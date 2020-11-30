import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from nflows import distributions
from pyprojroot import here

from src.data.toy import gen_bimodal_data
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

    X_samples = gen_bimodal_data(args.n_train)

    n_features = 1

    # plot data samples
    plot_hist(
        X_samples,
        color="blue",
        label="Real Data",
        save=str(save_path.joinpath("real_data.png")),
    )

    # get marginal transformation
    mg_trans = get_marginalization_transform(
        n_features=n_features, squash=args.squash, num_bins=args.n_bins
    )

    # initialize NF trainer
    gf_model = Gaussianization1D(
        mg_trans, base_distribution=distributions.StandardNormal(shape=[n_features])
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
    # ======================
    # Training Params
    # ======================
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning Rate",
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

    args = parser.parse_args()

    if args.smoke_test:
        os.environ["WANDB_MODE"] = "dryrun"
        args.epochs = 5
        args.n_train = 100
    main(args)
