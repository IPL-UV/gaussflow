from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from src.losses.utils import stable_avg_log_probs


class PlaneFlow(pl.LightningModule):
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

        log_pz = self.prior.log_prob(z).sum(dim=1)
        log_px = log_det_jac + log_pz
        return log_px

    @torch.no_grad()
    def predict_proba(self, x):

        z, log_det_jac = self.forward(x, rev=False)

        return self.log_prob(z, log_det_jac)

    @torch.no_grad()
    def sample(self, n_samples: int = 100, sample_shape=None):
        """
        Sample a batch of images from the flow.
        """
        # Sample latent representation from prior

        if sample_shape is not None:
            z = self.prior.sample(sample_shape=sample_shape).to(self.device)
        else:
            z = self.prior.sample(sample_shape=(n_samples, self.cfg.n_features)).to(
                self.device
            )

        # Transform z to x by inverting the flows
        z, _ = self.model(z, rev=True)

        return z

    def training_step(self, batch, batch_idx):

        # pass to INN and get transformed variable z and log Jacobian determinant
        z, log_det_jac = self.model(batch, rev=False)

        # calculate the negative log-likelihood of the model with a standard normal prior
        loss = self.loss_nll(z, log_det_jac)

        # log loss
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        # pass to INN and get transformed variable z and log Jacobian determinant
        z, log_det_jac = self.model(batch, rev=False)

        # calculate the negative log-likelihood of the model with a standard normal prior
        loss = self.loss_nll(z, log_det_jac)

        self.log("valid_loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):

        img_log_prob = []

        for _ in range(self.importance_samples):
            # pass to INN and get transformed variable z and log Jacobian determinant
            z, log_det_jac = self.model(batch, rev=False)

            # calculate the negative log-likelihood of the model with a standard normal prior
            img_log_prob.append(self.log_prob(z, log_det_jac))

        # stack images together
        img_log_prob = torch.stack(img_log_prob, dim=-1)

        img_log_prob = stable_avg_log_probs(img_log_prob)

        loss = -img_log_prob.mean()

        self.log("test_loss", loss)

    def loss_inn(self, z, log_det_jac):
        loss = 0.5 * torch.sum(z ** 2, 1) - log_det_jac
        loss = loss.mean() / z.shape[1]
        return loss

    def loss_nll(self, z, log_det_jac):
        loss = self.log_prob(z, log_det_jac)
        return -loss.mean()

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # optimizer settings
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--lr_scheduler", type=str, default="cosine_annealing")
        parser.add_argument("--num_epochs", type=int, default=100)
        # testing
        parser.add_argument("--importance_samples", type=int, default=8)
        parser.add_argument("--temperature", type=float, default=1.0)
        return parser


# class CondFlowLearnerPlane(FlowLearnerPlane):
#     def __init__(self, model: torch.nn.Module, base_dist, cfg):
#         super().__init__(model, base_dist, cfg)

#     def training_step(self, batch, batch_idx):

#         x, y = batch[0], batch[1]

#         # pass to INN and get transformed variable z and log Jacobian determinant
#         z, log_jac_det = self.model(x, c=[y])

#         # calculate the negative log-likelihood of the model with a standard normal prior
#         if self.cfg.loss_fn == "inn":
#             loss = 0.5 * torch.sum(z ** 2, 1) - log_jac_det
#             loss = loss.mean() / z.shape[1]
#         else:
#             loss = self.base_dist.log_prob(z).sum(1) + log_jac_det
#             loss = -loss.mean()

#         self.log("train_loss", loss)

#         return loss

#     def validation_step(self, batch, batch_idx):

#         x, y = batch[0], batch[1]

#         # pass to INN and get transformed variable z and log Jacobian determinant
#         z, log_jac_det = self.model(x, c=[y])

#         # calculate the negative log-likelihood of the model with a standard normal prior
#         if self.cfg.loss_fn == "inn":
#             loss = 0.5 * torch.sum(z ** 2, 1) - log_jac_det
#             loss = loss.mean() / z.shape[1]
#         else:
#             loss = self.base_dist.log_prob(z).sum(1) + log_jac_det
#             loss = -loss.mean()

#         self.log("valid_loss", loss)

#         return loss
