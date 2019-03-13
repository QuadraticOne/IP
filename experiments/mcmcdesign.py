from maths.activations import logistic
from maths.mcmc import metropolis_hastings


def f(x):
    """
    [Float] -> Float
    A test for a differentiable objective function.
    """
    return logistic(x[0])


def g(x):
    """
    [Float] -> Float
    A test for a non-differentiable objective function.
    """
    return 1e-8 if x[0] < 1 else 1


def run(n):
    """
    Int -> ()
    Attempt to optimise the parameters of a design by using the
    Metropolis-Hastings algorithm, for a certain number of iterations,
    to sample from the solution space that is most likely to satisfy
    the objective function.

    On testing, it failed to consistently find a good solution to
    even a simple objective function, regardless of whether or not
    it had a smooth derivative.
    """
    y = metropolis_hastings(n, g, [0.0])
    print(y, g(y))
