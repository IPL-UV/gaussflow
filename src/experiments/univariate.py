from src.data.toy import gen_bimodal_data
from src.viz.univariate import plot_hist
from argparse import ArgumentParser
from pathlib import Path
from src.models.gaussianization import get_marginalization_transform
import os
from pyprojroot import here

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

root = here(project_files=[".here"])
import wandb

import tqdm

home = str(Path.home())

save_path = Path(root).joinpath("reports/figures/experiments/univariate")


def main(args):

    X_samples = gen_bimodal_data(args.n_train)

    # plot data samples
    plot_hist(
        X_samples,
        color="blue",
        label="Real Data",
        save=str(save_path.joinpath("real_data.png")),
    )

    # get marginal transformation
    mg_trans = get_marginalization_transform(
        n_features=1, squash=args.squash, num_bins=args.n_bins
    )

    # base distribution
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
