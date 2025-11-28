from utils import *
import argparse


def test(
    data_root: str,
    batch_size: int = 8,
    weights = "",
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

    total_loss = 0.0
    total_pcr = 0.0
    total_mae = 0.0
    n_samples = 0

    # per salvare i singoli bag e trovare i 5 migliori/peggiori
    per_sample_records = []  # ogni entry: {"idx": int, "mae": float, "pred": tensor, "gt": tensor}

    with torch.no_grad():
        global_idx = 0
        for batch in val_loader:
            X, z = batch  # X=[B,121,16,16], z=[B,K]

            X = X.to(device)
            z = z.to(device)

            logits = model(X)  # [B,256,K]
            loss = llp_kl_bag_loss(logits, z)  # KL bag-loss

            probs = F.softmax(logits, dim=-1)  # [B,256,K]
            bag_pred = probs.mean(dim=1)  # [B,K]

            pcr = model.compute_pcr(z, bag_pred)
            mae_batch = (bag_pred - z).abs().mean()

            B = X.size(0)
            n_samples += B

            total_loss += loss.item() * B
            total_pcr += pcr.item() * B
            total_mae += mae_batch.item() * B

            # errore per singolo bag: MAE per riga
            per_bag_mae = (bag_pred - z).abs().mean(dim=1)  # [B]

            for i in range(B):
                # numero di farine presenti nel GT (stesso criterio di compute_pcr: z > 0)
                n_present = (z[i] > 0).sum().item()

                per_sample_records.append({
                    "idx": global_idx,
                    "mae": per_bag_mae[i].item(),
                    "pred": bag_pred[i].detach().cpu(),
                    "gt": z[i].detach().cpu(),
                    "n_present": n_present,  # <--- aggiunto
                })
                global_idx += 1

    # ----- metriche globali -----
    mean_loss = total_loss / n_samples
    mean_pcr = total_pcr / n_samples
    mean_mae = total_mae / n_samples

    print("\n=== RISULTATI VALIDAZIONE ===")
    print(f"KL bag-loss media : {mean_loss:.6f}")
    print(f"PCR medio         : {mean_pcr:.4f}")
    print(f"MAE medio         : {mean_mae:.6f}")  # se z è in [0,1], moltiplica per 100 per punti %

    # ============================
    # 5 migliori e 5 peggiori bag con DUE miscele presenti
    # ============================
    # filtro solo i bag dove il GT ha esattamente 2 farine presenti
    two_mix_records = [r for r in per_sample_records if r["n_present"] == 2]

    if len(two_mix_records) == 0:
        print("\n[ATTENZIONE] Nessun bag con esattamente 2 miscele trovate nel validation set.")
        return

    two_mix_records.sort(key=lambda d: d["mae"])  # crescente per MAE

    best_k = two_mix_records[:15]
    worst_k = two_mix_records[-5:] if len(two_mix_records) >= 5 else two_mix_records[-len(two_mix_records):]
    worst_k = list(reversed(worst_k))  # prima i peggiori veri

    def tensor_to_str(t):
        return "[" + ", ".join(f"{v:.3f}" for v in t.tolist()) + "]"

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
    arg.add_argument("--model_weight", type=str, required=True)
    arg.add_argument("--seed", type=int, default=42)

    args = arg.parse_args()

    test(
        data_root=args.data_root,
        batch_size=args.batch_size,
        weights=args.model_weight,
        seed=args.seed,
    )
