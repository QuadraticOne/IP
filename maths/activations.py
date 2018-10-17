from math import exp


def logistic(x):
    """
    Float -> Float
    """
    return 1 / (1 + exp(-x))


def identity(x):
    """
    a -> a
    The identity function; returns its input without changing it.
    """
    return x
