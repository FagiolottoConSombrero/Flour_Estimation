from utils import *
import argparse
import torch
import torch.nn.functional as F
from collections import defaultdict

FLOUR_ORDER = ["T", "E", "C", "A", "Z"]  # wheat, spelt, rye, oat, Z


def project_top2(p: torch.Tensor, eps: float = 1e-12):
    # p: [B,K] (probabilità)
    top2 = torch.topk(p, k=2, dim=1).indices            # [B,2]
    mask = torch.zeros_like(p).scatter(1, top2, 1.0)    # [B,K]
    p2 = p * mask
    p2 = p2 / (p2.sum(dim=1, keepdim=True).clamp_min(eps))
    return p2


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


def select_mid_examples(recs, metric_key="mae", mid_n=5, mode="median"):
    """
    Seleziona mid_n record "in mezzo":
    - mode="median": i più vicini alla mediana del metric_key
    - mode="mean":   i più vicini alla media del metric_key
    """
    vals = torch.tensor([r[metric_key] for r in recs], dtype=torch.float32)

    if mode == "mean":
        center = vals.mean().item()
    else:  # "median"
        center = vals.median().item()

    # ordina per distanza dal centro e prendi i primi mid_n
    recs_sorted = sorted(recs, key=lambda r: abs(r[metric_key] - center))
    return recs_sorted[:mid_n], center


def top2_set_hit(p_row: torch.Tensor, z_row: torch.Tensor) -> float:
    """
    ritorna 1.0 se il set delle top-2 classi predette coincide con il set delle top-2 del GT.
    (ordine non conta)
    """
    pred2 = torch.topk(p_row, k=2, dim=0).indices
    gt2   = torch.topk(z_row, k=2, dim=0).indices
    pred2, _ = torch.sort(pred2)
    gt2, _   = torch.sort(gt2)
    return float(torch.equal(pred2, gt2))


def pair_mass_from_gt(p_row: torch.Tensor, z_row: torch.Tensor) -> float:
    """
    somma delle probabilità predette sulle 2 classi GT (top-2 di z).
    range [0,1]. Più è alto, più il modello sta "puntando" sulle classi giuste.
    """
    gt2 = torch.topk(z_row, k=2, dim=0).indices
    return float(p_row[gt2].sum().item())


def test(
    data_root: str,
    batch_size: int = 8,
    weights: str = "",
    seed: int = 42,
    top_n: int = 5,
    worst_n: int = 5,
    mid_n: int = 5,
    mid_mode: str = "median",   # "median" oppure "mean"
    sort_metric: str = "mae"    # "mae" oppure "kld" (per ordinare best/worst)
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

            logits = model(X)  # tipicamente [B,?,K] dipende dal modello

            # ---- pred per bag ----
            bag_pred = F.softmax(logits, dim=-1)          # [B,K]
            bag_pred_top2 = project_top2(bag_pred)

            # ---- loss batch ----
            loss = llp_kl_patch_loss(logits, z)    # scalare batch

            # ---- metriche batch (MAE con top2) ----
            mae_batch = (bag_pred_top2 - z).abs().mean()

            B = X.size(0)
            n_samples += B
            total_loss += loss.item() * B
            total_mae += mae_batch.item() * B

            # ---- per-bag MAE e KLD ----
            per_bag_mae = (bag_pred - z).abs().mean(dim=1)              # [B]
            per_bag_kld = kld_per_bag_from_probs(bag_pred, z)           # [B]

            for i in range(B):
                top2_hit = None
                pair_mass = None

                n_present = (z[i] > 0).sum().item()
                if n_present == 2:
                    top2_hit = top2_set_hit(bag_pred[i].detach().cpu(), z[i].detach().cpu())
                    pair_mass = pair_mass_from_gt(bag_pred[i].detach().cpu(), z[i].detach().cpu())

                pair = None
                if n_present == 2:
                    pair = flour_pair_from_gt(z[i].detach().cpu(), FLOUR_ORDER)

                per_sample_records.append({
                    "idx": global_idx,
                    "mae": per_bag_mae[i].item(),
                    "kld": per_bag_kld[i].item(),
                    "pred": bag_pred[i].detach().cpu(),
                    "gt": z[i].detach().cpu(),
                    "n_present": n_present,
                    "pair": pair,
                    "top2_hit": top2_hit,
                    "pair_mass": pair_mass,
                })
                global_idx += 1

    # ----- metriche globali -----
    mean_loss = total_loss / n_samples
    mean_mae = total_mae / n_samples

    print("\n=== RISULTATI VALIDAZIONE ===")
    print(f"KL (patch-loss) media : {mean_loss:.6f}")
    print(f"MAE medio             : {mean_mae:.6f}")

    # ============================
    # TOP / MID / WORST per coppia (solo n_present==2)
    # ============================
    two_mix_records = [r for r in per_sample_records if r["n_present"] == 2 and r["pair"] is not None]

    top2_acc = sum(r["top2_hit"] for r in two_mix_records) / len(two_mix_records)
    pair_mass_mean = sum(r["pair_mass"] for r in two_mix_records) / len(two_mix_records)

    print("\n=== METRICHE TOP-2 (solo bag con 2 farine) ===")
    print(f"Top2-set accuracy (becco le 2 classi): {top2_acc:.4f}")
    print(f"PairMass media (prob sulle 2 classi GT): {pair_mass_mean:.4f}")

    # opzionale: quanto spesso PairMass supera una soglia
    thr = 0.90
    pm_at = sum(1 for r in two_mix_records if r["pair_mass"] >= thr) / len(two_mix_records)
    print(f"PairMass@{thr:.2f}: {pm_at:.4f}")

    if len(two_mix_records) == 0:
        print("\n[ATTENZIONE] Nessun bag con esattamente 2 miscele trovate nel validation set.")
        return

    by_pair = defaultdict(list)
    for r in two_mix_records:
        by_pair[r["pair"]].append(r)

    def tensor_to_str(t):
        return "[" + ", ".join(f"{v:.3f}" for v in t.tolist()) + "]"

    print("\n=== MIGLIORI / NELLA MEDIA / PEGGIORI PER COPPIA (solo bag con 2 farine) ===")
    for pair, recs in sorted(by_pair.items(), key=lambda x: x[0]):
        pair_top2_acc = sum(r["top2_hit"] for r in recs) / len(recs)
        pair_pm_mean = sum(r["pair_mass"] for r in recs) / len(recs)

        print(f"Top2-set acc (coppia): {pair_top2_acc:.4f} | PairMass mean: {pair_pm_mean:.4f}")

        # best/worst ordinati secondo sort_metric
        recs_sorted = sorted(recs, key=lambda d: d[sort_metric])

        best = recs_sorted[:top_n]
        worst = list(reversed(recs_sorted[-worst_n:]))

        # mid: vicini a mediana (o media) della metrica scelta
        mid, center = select_mid_examples(recs, metric_key=sort_metric, mid_n=mid_n, mode=mid_mode)

        print(f"\n\n############################")
        print(f"COPPIA {pair}  (N={len(recs)})")
        print(f"############################")
        print(f"Centro ({mid_mode}) su {sort_metric}: {center:.6f}")

        print(f"\n--- TOP-{top_n} ({sort_metric} più basso) ---")
        for r in best:
            print(f"\nIdx: {r['idx']} | MAE: {r['mae']:.6f} | KLD: {r['kld']:.6f}")
            print(f"  pred: {tensor_to_str(r['pred'])}")
            print(f"  gt  : {tensor_to_str(r['gt'])}")

        print(f"\n--- MID-{mid_n} (più vicini a {mid_mode} di {sort_metric}) ---")
        for r in mid:
            print(f"\nIdx: {r['idx']} | MAE: {r['mae']:.6f} | KLD: {r['kld']:.6f}")
            print(f"  pred: {tensor_to_str(r['pred'])}")
            print(f"  gt  : {tensor_to_str(r['gt'])}")

        print(f"\n--- WORST-{worst_n} ({sort_metric} più alto) ---")
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

    # nuove opzioni
    arg.add_argument("--top_n", type=int, default=5)
    arg.add_argument("--worst_n", type=int, default=5)
    arg.add_argument("--mid_n", type=int, default=5)
    arg.add_argument("--mid_mode", type=str, default="median", choices=["median", "mean"])
    arg.add_argument("--sort_metric", type=str, default="mae", choices=["mae", "kld"])

    args = arg.parse_args()

    test(
        data_root=args.data_root,
        batch_size=args.batch_size,
        weights=args.model_weight,
        seed=args.seed,
        top_n=args.top_n,
        worst_n=args.worst_n,
        mid_n=args.mid_n,
        mid_mode=args.mid_mode,
        sort_metric=args.sort_metric,
    )
