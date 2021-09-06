from argparse import ArgumentParser
from src.losses.utils import stable_avg_log_probs

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from nflows.utils import sum_except_batch
import numpy as np


class ImageFlow(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, cfg, prior=None):
        super().__init__()
        self.model = model
        self.cfg = cfg
        # Create prior distribution for final latent space
        self.prior = (
            torch.distributions.normal.Normal(loc=0.0, scale=1.0)
            if prior is None
            else prior
        )

    def forward(self, x, rev=False):
        return self.model(x, rev=rev)

    def log_prob(self, z: torch.Tensor, log_det_jac: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of images, return the likelihood of those.
        If return_ll is True, this function returns the log likelihood of the input.
        Otherwise, the ouput metric is bits per dimension (scaled negative log likelihood)
        """

        log_pz = self.prior.log_prob(z)
        log_pz = sum_except_batch(log_pz)
        log_px = log_det_jac + log_pz
        return log_px

    def loss_inn(self, z, log_det_jac):
        loss = 0.5 * torch.sum(z ** 2, [1, 2, 3]) - log_det_jac
        loss = loss.mean() / z.shape[1]
        return loss

    def loss_nll(self, z, log_det_jac):
        loss = -self.log_prob(z, log_det_jac)
        return loss.mean()

    def loss_bpd(self, z, log_det_jac):
        # calculate standard nll
        nll = -self.log_prob(z, log_det_jac)
        # scale loss by bits-per-dim
        loss = nll * np.log2(np.exp(1)) / np.prod(z.shape[1:])

        return loss.mean()

    @torch.no_grad()
    def predict_proba(self, x):

        z, log_det_jac = self.forward(x, rev=False)

        return self.log_prob(z, log_det_jac)

    @torch.no_grad()
    def sample(self, img_shape):
        """
        Sample a batch of images from the flow.
        """
        # Sample latent representation from prior

        z = self.prior.sample(sample_shape=img_shape).to(self.device)

        # Transform z to x by inverting the flows
        z, _ = self.model(z, rev=True)

        return z

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        # An scheduler is optional, but can help in flows to get the last bpd improvement
        if self.cfg.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 1, gamma=self.cfg.gamma
            )
        elif self.cfg.lr_scheduler == "reduce_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, self.cfg.n_total_steps, 0
            )

        elif self.cfg.lr_scheduler == "cosine_annealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.cfg.n_total_steps, 0
            )
        else:
            scheduler = None

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # segment batch
        x = batch

        # Normalizing flows are trained by maximum likelihood => return bpd
        z, log_det_jac = self.model(x)

        # get loss (bpd)
        loss = self.loss_bpd(z, log_det_jac)

        self.log("train_bpd", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # segment batch
        x = batch

        # Normalizing flows are trained by maximum likelihood => return bpd
        z, log_det_jac = self.model(x)

        # get loss (bpd)
        loss = self.loss_bpd(z, log_det_jac)

        self.log("val_bpd", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):

        img_log_prob = []

        for _ in range(self.cfg.importance_samples):
            # pass to INN and get transformed variable z and log Jacobian determinant
            z, log_det_jac = self.model(batch, rev=False)

            # calculate the negative log-likelihood of the model with a standard normal prior
            img_log_prob.append(self.log_prob(z, log_det_jac))

        # stack images together
        img_log_prob = torch.stack(img_log_prob, dim=-1)

        img_log_prob = stable_avg_log_probs(img_log_prob)
        # Calculate final bpd
        bpd = -img_log_prob * np.log2(np.exp(1)) / np.prod(batch[0].shape[1:])
        bpd = bpd.mean()

        self.log("test_bpd", bpd)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        #
        parser.add_argument("--multiscale", type=bool, default=True)
        # loss function
        parser.add_argument("--loss_fn", type=str, default="nll")
        # optimizer settings
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--lr_scheduler", type=str, default="cosine_annealing")
        parser.add_argument("--num_epochs", type=int, default=1_000)
        # testing
        parser.add_argument("--importance_samples", type=int, default=8)
        parser.add_argument("--temperature", type=float, default=1.0)
        return parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ImageFlow.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = MNIST("", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST("", train=False, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=args.batch_size)
    val_loader = DataLoader(mnist_val, batch_size=args.batch_size)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    model = ImageFlow(args.hidden_dim, args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    test_result = trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    cli_main()
