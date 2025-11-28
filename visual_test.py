import argparse
import matplotlib.pyplot as plt

from utils import *  # deve contenere: set_seed, make_llp_loaders, LLP, llp_kl_bag_loss


FLOUR_ORDER = ["T", "E", "C", "A", "Z"]  # wheat, spelt, rye, oat, Z
NUM_CLASSES = len(FLOUR_ORDER)


def load_model(weights: str, device: torch.device):
    """
    Carica il LightningModule LLP dal checkpoint Lightning.
    Coerente con il training:
        model = LLP(lr=1e-3, num_classes=5, patience=patience_loss)
    """
    model = LLP.load_from_checkpoint(weights)
    model.to(device)
    model.eval()
    return model


def tensor_to_str(t: torch.Tensor) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in t.tolist()) + "]"


def test(
    data_root: str,
    batch_size: int = 8,
    save_img: str = None,
    weights: str = None,
    seed: int = 42
):
    assert weights != "", "Devi passare il path ai pesi del modello (--model_weight)."

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    # ----- dataloader -----
    _, val_loader = make_llp_loaders(
        data_root=data_root,
        batch_size=batch_size,
        val_ratio=0.2
    )

    # ----- modello (LightningModule) -----
    model = load_model(weights, device)

    total_loss = 0.0
    total_pcr = 0.0
    total_mae = 0.0
    n_samples = 0

    # --- per MAE per classe ---
    class_abs_error_sum = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    class_count = 0  # numero di bag

    # --- per MAE per coppie di farine (solo bag con n_present == 2) ---
    # chiave: (i, j) con i < j, valore: {"mae_sum": float, "count": int}
    pair_stats = {}

    # per salvare i singoli bag e trovare i 5 migliori/peggiori
    per_sample_records = []  # ogni entry: {"idx", "mae", "pred", "gt", "n_present"}

    with torch.no_grad():
        global_idx = 0
        for batch in val_loader:
            # come nel tuo LLP.step:
            X, z = batch  # X=[B,121,16,16], z=[B,K]

            X = X.to(device)
            z = z.to(device)

            # forward: LLP.forward -> self.model(X) -> HSILLPMLP
            logits = model(X)                     # [B,256,K]
            loss = llp_kl_bag_loss(logits, z)     # KL bag-loss

            # ---- predizione del bag ----
            probs = F.softmax(logits, dim=-1)     # [B,256,K]
            bag_pred = probs.mean(dim=1)          # [B,K]

            # ---- metriche ----
            pcr = model.compute_pcr(z, bag_pred)
            mae_batch = (bag_pred - z).abs().mean()

            B = X.size(0)
            n_samples += B

            total_loss += loss.item() * B
            total_pcr  += pcr.item() * B
            total_mae  += mae_batch.item() * B

            # ----- MAE per classe -----
            abs_diff = (bag_pred - z).abs()           # [B, K]
            class_abs_error_sum += abs_diff.sum(dim=0).detach().cpu()
            class_count += B

            # ----- MAE per coppie di farine (n_present == 2) -----
            present_mask = (z > 0)  # [B, K] booleano

            for b in range(B):
                present_indices = torch.nonzero(present_mask[b], as_tuple=False).view(-1).tolist()
                if len(present_indices) == 2:
                    i, j = sorted(present_indices)
                    key = (i, j)

                    mae_bag = abs_diff[b].mean().item()

                    if key not in pair_stats:
                        pair_stats[key] = {"mae_sum": 0.0, "count": 0}
                    pair_stats[key]["mae_sum"] += mae_bag
                    pair_stats[key]["count"] += 1

            # ----- salva errori per sample (per best/worst) -----
            per_bag_mae = abs_diff.mean(dim=1)  # [B]

            for i in range(B):
                n_present = (z[i] > 0).sum().item()
                per_sample_records.append({
                    "idx": global_idx,
                    "mae": per_bag_mae[i].item(),
                    "pred": bag_pred[i].detach().cpu(),
                    "gt": z[i].detach().cpu(),
                    "n_present": n_present,
                })
                global_idx += 1

    # ----- metriche globali -----
    mean_loss = total_loss / n_samples
    mean_pcr  = total_pcr  / n_samples
    mean_mae  = total_mae  / n_samples

    print("\n=== RISULTATI VALIDAZIONE ===")
    print(f"KL bag-loss media : {mean_loss:.6f}")
    print(f"PCR medio         : {mean_pcr:.4f}")
    print(f"MAE medio         : {mean_mae:.6f}")

    # ===== MAE per classe =====
    mae_per_class = class_abs_error_sum / class_count  # [K]

    print("\n=== MAE PER CLASSE ===")
    for idx, mae_c in enumerate(mae_per_class.tolist()):
        flour = FLOUR_ORDER[idx]
        print(f"Classe {idx} ({flour}): MAE = {mae_c:.6f}")

    # ===== MAE per coppie di farine (solo bag con 2 miscele) =====
    print("\n=== MAE PER COPPIE DI FARINE (n_present == 2) ===")
    pair_mae_list = []
    for (i, j), stats in pair_stats.items():
        if stats["count"] > 0:
            avg_mae = stats["mae_sum"] / stats["count"]
            pair_mae_list.append((i, j, avg_mae, stats["count"]))

    pair_mae_list.sort(key=lambda x: x[2])  # per MAE crescente

    for i, j, avg_mae, cnt in pair_mae_list:
        fi, fj = FLOUR_ORDER[i], FLOUR_ORDER[j]
        print(f"Coppia ({fi}, {fj}): MAE = {avg_mae:.6f} su {cnt} bag")

    # ===== BAR PLOT MAE PER CLASSE =====
    plt.figure(figsize=(6, 4))
    x = np.arange(NUM_CLASSES)
    plt.bar(x, mae_per_class.numpy())
    plt.xticks(x, FLOUR_ORDER)
    plt.ylabel("MAE")
    plt.title("MAE per classe (abbondanza)")
    plt.tight_layout()
    plt.savefig(f"{save_img}/mae_per_classe.png", dpi=300)
    plt.close()

    # ===== HEATMAP MAE PER COPPIE DI FARINE =====
    heatmap = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=float)

    for (i, j), stats in pair_stats.items():
        if stats["count"] > 0:
            avg_mae = stats["mae_sum"] / stats["count"]
            heatmap[i, j] = avg_mae
            heatmap[j, i] = avg_mae  # simmetrica

    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap, interpolation="nearest")
    plt.colorbar(label="MAE medio")

    plt.xticks(range(NUM_CLASSES), FLOUR_ORDER)
    plt.yticks(range(NUM_CLASSES), FLOUR_ORDER)
    plt.title("MAE medio per coppia di farine (n_present == 2)")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if heatmap[i, j] > 0:
                plt.text(j, i, f"{heatmap[i, j]:.2f}",
                         ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{save_img}/heatmap_per_classe.png", dpi=300)
    plt.close()

    # ============================
    # 5 migliori e 5 peggiori bag con DUE miscele presenti
    # ============================
    two_mix_records = [r for r in per_sample_records if r["n_present"] == 2]

    if len(two_mix_records) == 0:
        print("\n[ATTENZIONE] Nessun bag con esattamente 2 miscele trovate nel validation set.")
        return

    two_mix_records.sort(key=lambda d: d["mae"])  # crescente per MAE

    best_k = two_mix_records[:5]
    worst_k = two_mix_records[-5:] if len(two_mix_records) >= 5 else two_mix_records[-len(two_mix_records):]
    worst_k = list(reversed(worst_k))  # prima i peggiori veri

    print("\n--- 5 MIGLIORI BAG (MAE più basso, SOLO con 2 miscele presenti) ---")
    for rec in best_k:
        print(f"\nIdx globale: {rec['idx']}, MAE: {rec['mae']:.6f}, n_present: {rec['n_present']}")
        print(f"  pred: {tensor_to_str(rec['pred'])}")
        print(f"  gt  : {tensor_to_str(rec['gt'])}")

    print("\n--- 5 PEGGIORI BAG (MAE più alto, SOLO con 2 miscele presenti) ---")
    for rec in worst_k:
        print(f"\nIdx globale: {rec['idx']}, MAE: {rec['mae']:.6f}, n_present: {rec['n_present']}")
        print(f"  pred: {tensor_to_str(rec['pred'])}")
        print(f"  gt  : {tensor_to_str(rec['gt'])}")


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--data_root", type=str, required=True)
    arg.add_argument("--batch_size", type=int, default=8)
    arg.add_argument("--save_img", type=str, default="~/projects/matteo/Flour_Estimation/plots")
    arg.add_argument("--model_weight", type=str, required=True)
    arg.add_argument("--seed", type=int, default=42)

    args = arg.parse_args()

    test(
        data_root=args.data_root,
        batch_size=args.batch_size,
        save_path=args.save_img,
        weights=args.model_weight,
        seed=args.seed,
    )
