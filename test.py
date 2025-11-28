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

    model = HSILLPMLP()
    ckpt = torch.load(weights, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)

    model.eval()

    total_mse = 0.0
    total_mae = 0.0
    total_pcr_num = 0.0
    total_pcr_den_x = 0.0
    total_pcr_den_y = 0.0
    n_samples = 0

    # Per salvare errori per singola bag
    per_sample_records = []  # lista di dict: {"idx": int, "mae": float, "pred": tensor, "gt": tensor}

    with torch.no_grad():
        global_idx = 0  # indice progressivo dei sample nel validation set

        for batch in val_loader:
            # Adatta questa parte in base a come il tuo Dataset restituisce i batch
            # Es. (bags, bag_labels) oppure (inputs, labels, meta, ...)
            inputs, bag_labels = batch  # <--- cambia se necessario

            inputs = inputs.to(device)  # [B, ...]
            bag_labels = bag_labels.to(device)  # [B, C]

            outputs = model(inputs)  # [B, C]

            # metriche batch
            mse = mse_loss(outputs, bag_labels)
            mae = mae_metric(outputs, bag_labels)

            # PCR "globale" su tutto il dataset:
            # per calcolarlo correttamente, accumuliamo numeratore e denominatore
            B = outputs.size(0)
            n_samples += B

            # Aggiorno somme per MSE/MAE (ponderate sul batch)
            total_mse += mse.item() * B
            total_mae += mae.item() * B

            # ----- accumulo per PCR globale -----
            x = outputs.reshape(-1)
            y = bag_labels.reshape(-1)
            x_mean = x.mean()
            y_mean = y.mean()
            num = torch.sum((x - x_mean) * (y - y_mean))
            den_x = torch.sum((x - x_mean) ** 2)
            den_y = torch.sum((y - y_mean) ** 2)

            total_pcr_num += num.item()
            total_pcr_den_x += den_x.item()
            total_pcr_den_y += den_y.item()

            # ----- salva errori per sample (per best/worst) -----
            # errore per bag = MAE per riga
            per_bag_mae = torch.mean(torch.abs(outputs - bag_labels), dim=1)  # [B]

            for i in range(B):
                per_sample_records.append({
                    "idx": global_idx,
                    "mae": per_bag_mae[i].item(),
                    "pred": outputs[i].detach().cpu(),
                    "gt": bag_labels[i].detach().cpu(),
                })
                global_idx += 1

    # metriche finali
    mean_mse = total_mse / n_samples
    mean_mae = total_mae / n_samples

    eps = 1e-8
    mean_pcr = total_pcr_num / ((total_pcr_den_x * total_pcr_den_y) ** 0.5 + eps)

    print("\n=== RISULTATI VALIDAZIONE ===")
    print(f"MSE medio : {mean_mse:.6f}")
    print(f"MAE medio : {mean_mae:.6f}")  # se le label sono in [0,1], moltiplica per 100 per avere punti %
    print(f"PCR medio : {mean_pcr:.4f}")

    # ============================
    # 5 migliori e 5 peggiori sample
    # ============================
    per_sample_records.sort(key=lambda d: d["mae"])  # sort crescente per MAE

    best_k = per_sample_records[:5]
    worst_k = per_sample_records[-5:] if len(per_sample_records) >= 5 else per_sample_records[-len(per_sample_records):]
    worst_k = list(reversed(worst_k))  # metto prima i peggiori veri

    def tensor_to_str(t):
        # Rappresentazione compatta
        return "[" + ", ".join(f"{v:.3f}" for v in t.tolist()) + "]"

    print("\n--- 5 MIGLIORI BAG (MAE più basso) ---")
    for rec in best_k:
        print(f"\nIdx globale: {rec['idx']}, MAE: {rec['mae']:.6f}")
        print(f"  pred: {tensor_to_str(rec['pred'])}")
        print(f"  gt  : {tensor_to_str(rec['gt'])}")

    print("\n--- 5 PEGGIORI BAG (MAE più alto) ---")
    for rec in worst_k:
        print(f"\nIdx globale: {rec['idx']}, MAE: {rec['mae']:.6f}")
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
