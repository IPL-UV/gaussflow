from argparse import ArgumentParser


def add_data_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    # data args
    parser.add_argument("--batch_size", type=int, default=128)
    return parser
