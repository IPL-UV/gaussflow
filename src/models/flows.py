import pytorch-lightning as pl
import torch

class NFLearner(pl.LightningModule):
    def __init__(self, model, hparams):
        super().__init__()
        self.model = model
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
            lr=self.hparams["lr"],
        )
