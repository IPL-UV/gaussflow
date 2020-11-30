import multiprocessing
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from nflows import distributions
from pyprojroot import here
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset

from src.data.toy import gen_bimodal_data
from src.models.dists import get_base_dist
from src.models.flows import Gaussianization1D
from src.models.gaussianization import get_marginalization_transform
from src.viz.univariate import plot_hist

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
root = here(project_files=[".here"])
import tqdm

import wandb

home = str(Path.home())

save_path = Path(root).joinpath("reports/figures/experiments/univariate")


def main(args):

    # =======================
    # Initialize Logger
    # =======================
    wandb_logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity)
    wandb_logger.experiment.config.update(args)

    seed_everything(args.seed)
    X_data = gen_bimodal_data(args.n_train)

    n_features = 1

    # plot data samples
    plot_hist(
        X_data,
        color="blue",
        label="Real Data",
        save=str(save_path.joinpath("samples_real.png")),
    )

    # get marginal transformation
    mg_trans = get_marginalization_transform(
        n_features=n_features,
        squash=args.squash,
        num_bins=args.n_bins,
        tails=args.tails,
        tail_bound=args.tail_bound,
        identity_init=args.identity,
    )

    # initialize NF trainer
    gf_model = Gaussianization1D(
        mg_trans, base_distribution=get_base_dist(n_features), hparams=args
    )

    # plot initial latent space
    with torch.no_grad():
        z = gf_model.model.transform_to_noise(torch.Tensor(X_data))
        plot_hist(
            z.numpy(),
            color="green",
            label="Latent Space",
            save=str(save_path.joinpath("latent_init.png")),
        )

    # ====================================
    # DATA
    # ====================================
    X_data = torch.FloatTensor(X_data)

    train_dataset = TensorDataset(X_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
    )

    # ====================================
    # TRAINING
    # ====================================
    trainer = Trainer(max_epochs=args.n_epochs, gpus=1, logger=wandb_logger)
    trainer.fit(gf_model, train_loader)

    # ====================================
    # PLOTS
    # ====================================
    with torch.no_grad():
        z = gf_model.model.transform_to_noise(X_data)
        plot_hist(
            z.detach().numpy(),
            color="green",
            label="Latent Space",
            save=str(save_path.joinpath("latent_trained.png")),
        )

    # SAMPLING
    with torch.no_grad():
        X_approx = gf_model.model.sample(args.n_samples)
        plot_hist(
            X_approx.numpy(),
            color="red",
            label="Gen. Samples",
            save=str(save_path.joinpath("samples_gen.png")),
        )
    pass


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    # ======================
    # Data parameters
    # ======================
    parser.add_argument(
        "--n-train",
        type=int,
        default=2_000,
        help="Number of training samples",
    )
    # ======================
    # Transform Params
    # ======================
    parser.add_argument(
        "--squash",
        type=int,
        default=0,
        help="Number of bins for spline transformation",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of bins for spline transformation",
    )
    parser.add_argument(
        "--tail-bound",
        type=float,
        default=10.0,
        help="Number of bins for spline transformation",
    )
    parser.add_argument(
        "--tails",
        type=str,
        default="linear",
        help="tails",
    )
    parser.add_argument(
        "--identity",
        type=bool,
        default=1,
        help="Initialize with identity",
    )
    # ======================
    # Training Params
    # ======================
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning Rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=100,
        help="Number of epochs for training",
    )
    # ======================
    # VIZ Params
    # ======================
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5_000,
        help="Number of samples",
    )
    # ======================
    # Testing
    # ======================
    parser.add_argument(
        "-sm",
        "--smoke-test",
        action="store_true",
        help="to do a smoke test without logging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for project",
    )
    # ======================
    # Logger Parameters
    # ======================
    parser.add_argument("--wandb-entity", type=str, default="emanjohnson91")
    parser.add_argument("--wandb-project", type=str, default="rbig20-1d")

    args = parser.parse_args()

    if args.smoke_test:
        os.environ["WANDB_MODE"] = "dryrun"
        args.n_epochs = 5
        args.n_train = 100
    main(args)
