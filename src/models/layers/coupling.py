import FrEIA.modules as Fm


def get_coupling_layer(coupling: str = "glow"):
    # coupling transform (GLOW)
    if coupling == "glow":
        coupling_transform = Fm.GLOWCouplingBlock
    elif coupling == "realnvp":
        coupling_transform = Fm.RNVPCouplingBlock
    elif coupling == "nice":
        coupling_transform = Fm.NICECouplingBlock
    else:
        raise ValueError(f"unrecognized coupling transform: {coupling}")

    return coupling_transform