from argparse import ArgumentParser


def add_wandb_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    # data args
    parser.add_argument("--wandb_project", type=str, default="cvpr2021")
    parser.add_argument("--wandb_entity", type=str, default="ipl_uv")
    return parser
