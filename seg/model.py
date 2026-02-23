import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from .loss import SoftCLDiceLoss


class MInterface(pl.LightningModule):

    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
        )

        self.loss_fn = SoftCLDiceLoss()

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        return loss, probs, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
