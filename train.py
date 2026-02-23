import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from seg.model import MInterface
from seg.data import DInterface


def main():
    pl.seed_everything(42)

    data_module = DInterface(
        data_dir=DATA_DIR,
        set_id=SET_ID,
        batch_size=BATCH_SIZE,
    )

    model = MInterface(lr=LR)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.4f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    logger = TensorBoardLogger(
        save_dir=SAVE_DIR,
        name="segmentation",
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
    )

    trainer.fit(model, data_module)

    print("\nDone.")
    print("Best model path:", checkpoint_callback.best_model_path)


if __name__ == "__main__":
    DATA_DIR = "./Dataset"
    SET_ID = 2  # 1 or 2 for train
    BATCH_SIZE = 8
    EPOCHS = 20
    LR = 1e-3
    SAVE_DIR = f"checkpoints_set{SET_ID}"
    main()
