from argparse import ArgumentParser
import yaml


def add_wandb_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    # data args
    parser.add_argument("--wandb_project", type=str, default="cvpr2021")
    parser.add_argument("--wandb_entity", type=str, default="ipl_uv")
    return parser


def update_args_yaml(args, yaml_file: str):
    if yaml_file is not None:
        opt = yaml.load(open(yaml_file), Loader=yaml.FullLoader)
        opt.update(vars(args))
        args = opt
    return args
