import pytorch_lightning as pl
import torch
from nflows.flows import Flow
from pytorch_lightning import Trainer


class Gaussianization1D(pl.LightningModule):
    def __init__(self, transform, base_distribution, hparams):
        super().__init__()
        self.model = Flow(transform=transform, distribution=base_distribution)
        self.hparams = hparams

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # self.model.train()
        # self.likelihood.train()
        (x,) = batch

        # loss function: negative log-likelihood
        loss = -self.model.log_prob(inputs=x).mean()

        # log loss function
        self.log("loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            [
                {"params": self.model.parameters()},
            ],
            lr=self.hparams.lr,
        )


class Gaussianization2D(pl.LightningModule):
    def __init__(self, transform, base_distribution, hparams):
        super().__init__()
        self.model = Flow(transform=transform, distribution=base_distribution)
        self.hparams = hparams

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # self.model.train()
        # self.likelihood.train()
        (x,) = batch

        # loss function: negative log-likelihood
        loss = -self.model.log_prob(inputs=x).mean()

        # log loss function
        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        # self.model.train()
        # self.likelihood.train()
        (x,) = batch

        # loss function: negative log-likelihood
        loss = -self.model.log_prob(inputs=x).mean()

        # log loss function
        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            [
                {"params": self.model.parameters()},
            ],
            lr=self.hparams.lr,
        )
