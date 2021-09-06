from argparse import ArgumentParser
import yaml
import torch


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


def gf_propagate(inn, init_X):
    #     print("Before:", init_X.shape)
    with torch.no_grad():
        z, ldj = inn.module_list[-1](
            [
                init_X,
            ]
        )
    if isinstance(z, tuple):
        z = z[0]
    #     print("After:", z.shape)
    return z
