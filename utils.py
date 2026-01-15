import pytorch_lightning as pl
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from model import *
from dataloader import *
import pandas as pd


def llp_kl_bag_loss(logits, z, eps=1e-8):
    """
    logits: [B, P, K]  output del modello per tutti i pixel
    z:      [B, K]     vettori di abbondanza (proporzioni farine) per ciascun bag/patch
    """
    # Probabilità pixel-wise
    probs = F.softmax(logits, dim=-1)   # [B, P, K]
    # Media sui pixel → predizione di bag
    bag_pred = probs.mean(dim=1)        # [B, K]
    # Evita log(0)
    bag_pred = bag_pred.clamp(min=eps)
    # Cross-entropy con target soft (z): - sum z_k log(pred_k)
    loss_per_bag = -(z * bag_pred.log()).sum(dim=-1)   # [B]
    return loss_per_bag.mean()


def llp_kl_patch_loss(logits, z, eps=1e-8):
    """
    logits: [B, K]  output del modello per ciascun bag/patch
    z:      [B, K]  vettori di abbondanza (proporzioni farine) per ciascun bag/patch
    """
    probs = F.softmax(logits, dim=-1)   # [B, K]
    probs = probs.clamp(min=eps)
    # cross-entropy con target soft: -sum_k z_k log p_k
    loss_per_bag = -(z * probs.log()).sum(dim=-1)   # [B]
    return loss_per_bag.mean()


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- LightningModule ----------------
class LLP(pl.LightningModule):
    def __init__(self, lr=1e-3, num_classes=5, patience=20, model_type=1, rgb=False, ir=False):
        super().__init__()
        self.model_type = model_type
        self.rgb = rgb
        self.ir = ir
        self.save_hyperparameters()
        if model_type == 1:
            self.model = HSILLPMLP(in_bands=121, n_classes=num_classes)
        elif model_type == 2:
            self.model = HSILSpectralCNN(in_bands=121, n_classes=num_classes)
        elif model_type == 3:
            self.model = HSILLP_PatchCNN(in_bands=121, n_classes=num_classes)
        self.lr = lr
        self.num_classes = num_classes
        self.patience = patience

    def forward(self, x):  # x = radianza HSI: [B,121,16,16]
        if self.rgb:
            x = simulate_no_filter_camera(x)
        elif self.ir:
            x = simulate_no_filter_camera(x, use_ir=True)
        return self.model(x)  # [B,121,16,16]

    def compute_pcr(self, z, bag_pred, threshold=0.1):
        """
        Present Classes Recall (PCR)
        Valuta se il modello trova entrambe le farine presenti nella patch.
        """
        true_present = (z > 0)                # [B,K]
        pred_present = (bag_pred > threshold) # [B,K]

        correct = (true_present & pred_present).float().sum(dim=1)
        total_true = true_present.float().sum(dim=1)

        pcr = (correct / (total_true + 1e-8)).mean()
        return pcr

    def step(self, batch, stage):
        X, z = batch  # X=[B,121,16,16]  z=[B,K]

        logits = self(X)                     # [B,256,5]
        if self.model_type == 1 | self.model_type == 2:
            loss = llp_kl_bag_loss(logits, z)    # KL bag-loss
        else:
            loss = llp_kl_patch_loss(logits, z)

        # ---- predizione del bag ----
        bag_pred = F.softmax(logits, dim=-1)    # [B,256,K]
        #bag_pred = probs.mean(dim=1)         # [B,K]
        #eps = 1e-12
        #bag_pred = torch.sigmoid(logits)  # [B,256,K]
        #bag_pred = bag_pred / bag_pred.sum(dim=-1, keepdim=True).clamp_min(eps)
        #eps = 1e-12
        #bag_pred = F.relu(logits)  # [B,256,K]
        #bag_pred = bag_pred / bag_pred.sum(dim=-1, keepdim=True).clamp_min(eps)

        # ---- metriche ----
        #pcr = self.compute_pcr(z, bag_pred)  # Present Class Recall
        mae = (bag_pred - z).abs().mean()

        # ---- logging ----
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        #self.log(f"{stage}_pcr", pcr, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mae", mae, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=self.patience
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss"
            },
        }


# ----- dataloader con split automatico -----
def make_llp_loaders(data_root, batch_size=8, val_ratio=0.2):
    # carica l'intero dataset
    full_ds = HSILLPDataset(data_root)
    # generiamo gli indici
    indices = list(range(len(full_ds)))
    train_idx, val_idx = train_test_split(indices, test_size=val_ratio, shuffle=True, random_state=42)
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    return train_loader, val_loader


def load_model(weights: str, device: torch.device):
    """
    Carica il LightningModule LLP dal checkpoint Lightning.
    Questo è coerente con come hai addestrato il modello:
        model = LLP(lr=1e-3, num_classes=5, patience=patience_loss)
    """
    model = LLP.load_from_checkpoint(weights)  # hyperparameters vengono caricati dal ckpt
    model.to(device)
    model.eval()
    return model


def simulate_no_filter_camera(HSI, use_ir: bool = False, csv_path: str = '/home/acp/Documenti/Sony_ILCE_6100_RGBIR_scaled_005.csv'):
    """
    Simula acquisizione camera senza filtro (solo integrazione spettrale con curve del sensore).

    Args:
        HSI:    torch.Tensor [B,121,H,W]
        use_ir: se True usa RGB+IR (4 canali), altrimenti solo RGB (3 canali)
        csv_path: path al csv con le curve

    Returns:
        img: torch.Tensor [B,C,H,W] con C=4 (RGBIR) oppure C=3 (RGB)
    """
    curves = get_sensor_curves(csv_path=csv_path, use_ir=use_ir).to(HSI.device)  # [C,121]
    img = torch.einsum('b l h w, c l -> b c h w', HSI, curves)
    return img


def get_sensor_curves(
    csv_path: str = '/home/acp/Documenti/Sony_ILCE_6100_RGBIR_scaled_005.csv',
    use_ir: bool = True,
    ir_column: str = "IR850",
    rgb_columns=("red", "green", "blue"),
):
    """
    Legge le curve spettrali del sensore e restituisce un tensore PyTorch.

    Args:
        csv_path: path csv
        use_ir: se True include IR (4 canali), altrimenti solo RGB (3 canali)
        ir_column: nome colonna IR nel csv
        rgb_columns: nomi colonne RGB nel csv

    Returns:
        curves: torch.Tensor [C,121] con ordine (R,G,B[,IR])
    """
    df = pd.read_csv(csv_path)

    # --- 1) Leggi colonne ---
    r_col, g_col, b_col = rgb_columns
    R = torch.tensor(df[r_col].values, dtype=torch.float32).numpy()
    G = torch.tensor(df[g_col].values, dtype=torch.float32).numpy()
    B = torch.tensor(df[b_col].values, dtype=torch.float32).numpy()

    if use_ir:
        if ir_column not in df.columns:
            raise KeyError(f"Colonna IR '{ir_column}' non trovata nel CSV. Colonne disponibili: {list(df.columns)}")
        IR = torch.tensor(df[ir_column].values, dtype=torch.float32).numpy()

    # --- 2) Assi spettrali ---
    wavelength_5nm = np.arange(400, 1000 + 1, 5)    # 121 valori
    wavelength_10nm = np.arange(400, 1000 + 1, 10)  # tipico input csv (61 valori)

    # --- 3) Interpolazione a 5 nm ---
    R_5 = np.interp(wavelength_5nm, wavelength_10nm, R)
    G_5 = np.interp(wavelength_5nm, wavelength_10nm, G)
    B_5 = np.interp(wavelength_5nm, wavelength_10nm, B)

    curves_list = [R_5, G_5, B_5]

    if use_ir:
        IR_5 = np.interp(wavelength_5nm, wavelength_10nm, IR)
        curves_list.append(IR_5)

    # --- 4) Stack in [C,121] ---
    curves = torch.tensor(np.stack(curves_list, axis=0), dtype=torch.float32)

    return curves
