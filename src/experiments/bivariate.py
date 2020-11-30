import multiprocessing
import os
from argparse import ArgumentParser
from pathlib import Path
from nflows import transforms

import torch
from nflows import distributions
from pyprojroot import here
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset

from src.data.toy import get_bivariate_data
from src.models.dists import get_base_dist
from src.models.flows import Gaussianization2D
from src.models.gaussianization import get_marginalization_transform, get_rotation
from src.viz.bivariate import plot_2d_joint, plot_2d_joint_probs

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
root = here(project_files=[".here"])
home = str(Path.home())

save_path = Path(root).joinpath("reports/figures/experiments/bivariate")


def main(args):

    # =======================
    # Initialize Logger
    # =======================
    wandb_logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity)
    wandb_logger.experiment.config.update(args)

    seed_everything(args.seed)
    X_data = get_bivariate_data(
        dataset=args.dataset, n_samples=args.n_train, noise=args.noise, seed=args.seed
    )
    X_val = get_bivariate_data(
        dataset=args.dataset,
        n_samples=args.n_valid,
        noise=args.noise,
        seed=args.seed + 1,
    )

    n_features = 2

    # plot data samples
    plot_2d_joint(
        X_data,
        color="blue",
        label="Real Data",
        wandb_logger=wandb_logger.experiment,
        log_name="samples_real",
        # save=str(save_path.joinpath(f"{args.dataset}_samples_real.png")),
    )

    # get number of layers
    layers = []
    if args.init_rot:
        # initialize with rotation layer
        layers.append(
            get_rotation(
                n_features=n_features,
                num_householder=args.num_householder,
                identity_init=args.identity,
                rotation=args.rotation,
            )
        )

    # loop through layers
    for _ in range(args.n_layers):

        # marginal transform
        layers.append(
            get_marginalization_transform(
                n_features=n_features,
                squash=args.squash,
                num_bins=args.n_bins,
                tails=args.tails,
                tail_bound=args.tail_bound,
                identity_init=args.identity,
            )
        )
        # rotation
        layers.append(
            get_rotation(
                n_features=n_features,
                num_householder=args.num_householder,
                identity_init=args.identity,
                rotation=args.rotation,
            )
        )

    # get marginal transformation
    gauss_flows = transforms.CompositeTransform(layers)
    # createval_loader

    # initialize NF trainer
    gf_model = Gaussianization2D(
        gauss_flows, base_distribution=get_base_dist(n_features), hparams=args
    )

    # plot initial latent space
    with torch.no_grad():
        z = gf_model.model.transform_to_noise(torch.Tensor(X_data))
        plot_2d_joint(
            z.numpy(),
            color="green",
            label="Latent Space",
            wandb_logger=wandb_logger.experiment,
            log_name="latent_init",
            # save=str(save_path.joinpath(f"{args.dataset}_samples_real.png")),
        )

    # ====================================
    # DATA
    # ====================================
    X_data, X_val = torch.FloatTensor(X_data), torch.FloatTensor(X_val)

    train_dataset, val_dataset = TensorDataset(X_data), TensorDataset(X_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
    )

    # ====================================
    # TRAINING
    # ====================================
    trainer = Trainer(max_epochs=args.n_epochs, gpus=1, logger=wandb_logger)
    trainer.fit(gf_model, train_loader, val_loader)

    # ====================================
    # PLOTS
    # ====================================
    with torch.no_grad():
        # LATENT SPACE
        z = gf_model.model.transform_to_noise(X_data)
        plot_2d_joint(
            z.detach().numpy(),
            color="green",
            label="Latent Space",
            wandb_logger=wandb_logger.experiment,
            log_name="latent_trained",
            # save=str(save_path.joinpath("latent_trained.png")),
        )

        # PROBABILITIES
        X_logprob = gf_model.model.log_prob(X_data)

        plot_2d_joint_probs(
            X_data.detach().numpy(),
            probs=X_logprob.numpy(),
            wandb_logger=wandb_logger.experiment,
            log_name="log_probs",
            # save=str(save_path.joinpath("latent_trained.png")),
        )
        plot_2d_joint_probs(
            X_data.detach().numpy(),
            probs=X_logprob.exp().numpy(),
            wandb_logger=wandb_logger.experiment,
            log_name="probs",
            # save=str(save_path.joinpath("latent_trained.png")),
        )

    # SAMPLING
    with torch.no_grad():
        X_approx = gf_model.model.sample(args.n_samples)
        plot_2d_joint(
            X_approx.numpy(),
            color="red",
            label="Gen. Samples",
            wandb_logger=wandb_logger.experiment,
            log_name="samples_gen",
            # save=str(save_path.joinpath("samples_gen.png")),
        )


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    # ======================
    # Data parameters
    # ======================
    parser.add_argument(
        "--dataset",
        type=str,
        default="rbig",
        help="2D Dataset",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=5_000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--n-valid",
        type=int,
        default=500,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.10,
        help="Noise level",
    )
    # ======================
    # Transform Params
    # ======================
    parser.add_argument(
        "--init-rot",
        type=int,
        default=1,
        help="Init rotation",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=3,
        help="Number of layers",
    )
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
        type=int,
        default=1,
        help="Initialize with identity",
    )
    parser.add_argument(
        "--rotation",
        type=str,
        default="pca",
        help="Rotation layer",
    )
    parser.add_argument(
        "--num-householder",
        type=int,
        default=2,
        help="Number of householder matrices",
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
    parser.add_argument("--wandb-project", type=str, default="rbig20-2d")

    args = parser.parse_args()

    if args.smoke_test:
        os.environ["WANDB_MODE"] = "dryrun"
        args.n_epochs = 5
        args.n_train = 100
    main(args)
