from utils import *
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import argparse


def main(
    data_root: str,
    save_dir: str = "runs/llp",
    model_type: int = 1,
    batch_size: int = 8,
    epochs: int = 50,
    seed: int = 42,
    patience_loss: int = 50,
    patience_es: int = 50,
    devices="auto",
):

    set_seed(seed)

    # ----- dataloader -----
    train_loader, val_loader = make_llp_loaders(
        data_root=data_root,
        batch_size=batch_size,
        val_ratio=0.2
    )

    # ----- modello LLP -----
    model = LLP(lr=1e-3, num_classes=5, patience=patience_loss, model_type=model_type)

    # ----- callbacks -----
    ckpt = ModelCheckpoint(dirpath=save_dir, filename="best",monitor="val_loss", mode="min", save_top_k=1)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=patience_es, verbose=True)
    lrmon = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        default_root_dir=save_dir,
        max_epochs=epochs,
        accelerator="auto",
        devices=devices,
        precision="32-true",
        callbacks=[ckpt, early, lrmon],
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


# ---------------- CLI ----------------
if __name__ == "__main__":

    arg = argparse.ArgumentParser()
    arg.add_argument("--data_root", type=str, required=True)
    arg.add_argument("--save_dir", type=str, default="runs/llp")
    arg.add_argument("--batch_size", type=int, default=8)
    arg.add_argument("--model_type", type=int, default=1)
    arg.add_argument("--epochs", type=int, default=1000)
    arg.add_argument("--seed", type=int, default=42)
    arg.add_argument("--patience_loss", type=int, default=50)
    arg.add_argument("--patience_early_stopping", type=int, default=100)
    arg.add_argument("--devices", type=str, default="auto")

    args = arg.parse_args()

    main(
        data_root=args.data_root,
        save_dir=args.save_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        patience_loss=args.patience_loss,
        patience_es=args.patience_early_stopping,
        devices=args.devices,
    )