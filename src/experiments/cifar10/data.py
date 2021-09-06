from argparse import ArgumentParser
import os


def add_cifar10_ds_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    # data args
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--dataset_dir", type=str, default="/media/disk/databases/CIFAR10"
    )
    parser.add_argument(
        "--num_workers", type=int, default=min(16, int(os.cpu_count() / 2))
    )
    return parser
