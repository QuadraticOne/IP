from math import exp


def logistic(x):
    """
    Float -> Float
    """
    return 1 / (1 + exp(x))
