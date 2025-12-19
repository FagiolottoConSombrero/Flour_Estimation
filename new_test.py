from utils import *
import argparse
import torch
import torch.nn.functional as F
from collections import defaultdict

FLOUR_ORDER = ["T", "E", "C", "A", "Z"]  # wheat, spelt, rye, oat, Z

def kld_per_bag_from_probs(bag_pred: torch.Tensor, z: torch.Tensor, eps: float = 1e-8):
    """
    bag_pred: [B,K] probabilità (somma=1)
    z      : [B,K] target (tipicamente somma=1)
    ritorna: kld per bag [B]
    KLD(z || bag_pred) = sum_k z_k * log(z_k / bag_pred_k)
    """
    z_ = z.clamp(min=eps)
    p_ = bag_pred.clamp(min=eps)
    return (z_ * (z_.log() - p_.log())).sum(dim=1)  # [B]


def flour_pair_from_gt(z_row: torch.Tensor, flour_order):
    """
    z_row: [K]
    ritorna tuple ordinata delle 2 farine presenti, es ('A','Z')
    """
    idxs = torch.nonzero(z_row > 0, as_tuple=False).squeeze(1).tolist()
    letters = [flour_order[i] for i in idxs]
    letters = sorted(letters)
    return tuple(letters)


def test(
    data_root: str,
    batch_size: int = 8,
    weights: str = "",
    seed: int = 42
):
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
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    n_samples = 0

    # salva record per campione
    per_sample_records = []  # {"idx","mae","kld","pred","gt","n_present","pair"}

    with torch.no_grad():
        global_idx = 0
        for batch in val_loader:
            X, z = batch  # X=[B,121,16,16], z=[B,K]
            X = X.to(device)
            z = z.to(device)

            logits = model(X)  # tipicamente [B,256,K]

            # ---- pred per bag (media pixel) ----
            bag_pred = F.softmax(logits, dim=-1)          # [B,K]

            # ---- loss batch (come prima) ----
            loss = llp_kl_patch_loss(logits, z)    # scalare batch

            # ---- metriche batch ----
            mae_batch = (bag_pred - z).abs().mean()

            B = X.size(0)
            n_samples += B
            total_loss += loss.item() * B
            total_mae += mae_batch.item() * B

            # ---- per-bag MAE e KLD ----
            per_bag_mae = (bag_pred - z).abs().mean(dim=1)              # [B]
            per_bag_kld = kld_per_bag_from_probs(bag_pred, z)           # [B]

            for i in range(B):
                n_present = (z[i] > 0).sum().item()

                pair = None
                if n_present == 2:
                    # usa l'ordine farine definito in utils.py (FLOUR_ORDER)
                    pair = flour_pair_from_gt(z[i].detach().cpu(), FLOUR_ORDER)

                per_sample_records.append({
                    "idx": global_idx,
                    "mae": per_bag_mae[i].item(),
                    "kld": per_bag_kld[i].item(),
                    "pred": bag_pred[i].detach().cpu(),
                    "gt": z[i].detach().cpu(),
                    "n_present": n_present,
                    "pair": pair,
                })
                global_idx += 1

    # ----- metriche globali -----
    mean_loss = total_loss / n_samples
    mean_mae = total_mae / n_samples

    print("\n=== RISULTATI VALIDAZIONE ===")
    print(f"KL (patch-loss) media : {mean_loss:.6f}")
    print(f"MAE medio             : {mean_mae:.6f}")

    # ============================
    # TOP-5 e WORST-5 per ogni coppia (solo n_present==2)
    # ============================
    two_mix_records = [r for r in per_sample_records if r["n_present"] == 2 and r["pair"] is not None]

    if len(two_mix_records) == 0:
        print("\n[ATTENZIONE] Nessun bag con esattamente 2 miscele trovate nel validation set.")
        return

    # raggruppa per coppia
    by_pair = defaultdict(list)
    for r in two_mix_records:
        by_pair[r["pair"]].append(r)

    def tensor_to_str(t):
        return "[" + ", ".join(f"{v:.3f}" for v in t.tolist()) + "]"

    print("\n=== MIGLIORI/PEGGIORI PER COPPIA (solo bag con 2 farine) ===")
    for pair, recs in sorted(by_pair.items(), key=lambda x: x[0]):
        recs_sorted = sorted(recs, key=lambda d: d["mae"])  # ordina per MAE (puoi cambiarlo in kld se vuoi)

        best = recs_sorted[:25]
        worst = recs_sorted[-5:]
        worst = list(reversed(worst))

        print(f"\n\n############################")
        print(f"COPPIA {pair}  (N={len(recs)})")
        print(f"############################")

        print("\n--- TOP-5 (MAE più basso) ---")
        for r in best:
            print(f"\nIdx: {r['idx']} | MAE: {r['mae']:.6f} | KLD: {r['kld']:.6f}")
            print(f"  pred: {tensor_to_str(r['pred'])}")
            print(f"  gt  : {tensor_to_str(r['gt'])}")

        print("\n--- WORST-5 (MAE più alto) ---")
        for r in worst:
            print(f"\nIdx: {r['idx']} | MAE: {r['mae']:.6f} | KLD: {r['kld']:.6f}")
            print(f"  pred: {tensor_to_str(r['pred'])}")
            print(f"  gt  : {tensor_to_str(r['gt'])}")


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--data_root", type=str, required=True)
    arg.add_argument("--batch_size", type=int, default=8)
    arg.add_argument("--model_weight", type=str, required=True)
    arg.add_argument("--seed", type=int, default=42)
    args = arg.parse_args()

    test(
        data_root=args.data_root,
        batch_size=args.batch_size,
        weights=args.model_weight,
        seed=args.seed,
    )
