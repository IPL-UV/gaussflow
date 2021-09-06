from src.experiments.cifar10.models.gf import (
    add_multiscale_cifar10_model_gf_simple_args,
    create_multiscale_cifar10_model_gf_simple,
)


def get_model_args(parser, model):

    if model == "glow":
        raise NotImplementedError()

    elif model == "gf":

        parser = add_multiscale_cifar10_model_gf_simple_args(parser)

    else:
        raise ValueError(f"Unrecognized model: {model}")

    return parser


def get_model_architecture(args, X_init, model="gf"):

    if model == "gf":

        model = create_multiscale_cifar10_model_gf_simple(
            X_init,
            n_subflows_1=args.n_subflows_1,
            n_subflows_2=args.n_subflows_2,
            n_subflows_3=args.n_subflows_3,
            n_subflows_4=args.n_subflows_4,
            n_reflections=args.n_reflections,
            n_components=args.n_components,
        )

    else:
        raise ValueError(f"Unrecognized model: {model}")
    return model
