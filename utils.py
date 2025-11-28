import pytorch_lightning as pl
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from model import *
from dataloader import *


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


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- LightningModule ----------------
class LLP(pl.LightningModule):

    def __init__(self, lr=1e-3, num_classes=5, patience=20):
        super().__init__()
        self.save_hyperparameters()
        self.model = HSILLPMLP(in_bands=121, n_classes=num_classes)
        self.lr = lr
        self.num_classes = num_classes
        self.patience = patience

    def forward(self, x):
        return self.model(x)   # [B,256,K]

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
        loss = llp_kl_bag_loss(logits, z)    # KL bag-loss

        # ---- predizione del bag ----
        probs = F.softmax(logits, dim=-1)    # [B,256,K]
        bag_pred = probs.mean(dim=1)         # [B,K]

        # ---- metriche ----
        pcr = self.compute_pcr(z, bag_pred)  # Present Class Recall
        mae = (bag_pred - z).abs().mean()

        # ---- logging ----
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_pcr", pcr, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mae", mae, on_epoch=True, prog_bar=False)

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