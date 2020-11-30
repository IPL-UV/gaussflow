from nflows import distributions


def get_base_dist(n_features: int):

    return distributions.StandardNormal(shape=[n_features])
