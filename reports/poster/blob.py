from math import cos, sin, pi
from numpy import linspace
from random import uniform
import matplotlib.pyplot as plt


def get_coordinates(radius_function, segments=100, offset=(0.0, 0.0)):
    """
    (Float -> Float) -> Int? -> ([Float], [Float])
    Get a set of coordinates for a radius function.
    """
    thetas = linspace(0, 2 * pi, segments)
    xs, ys = [], []
    x, y = offset
    for theta in thetas:
        r = radius_function(theta)
        xs.append(r * cos(theta) + x)
        ys.append(r * sin(theta) + y)
    return xs, ys


def plot_coordinates(radius_function, segments=100, alpha=0.4, offset=(0.0, 0.0)):
    """
    (Float -> Float) -> Int? -> ([Float], [Float])
    Plot a set of coordinates for a radius function.
    """
    xs, ys = get_coordinates(radius_function, segments=segments, offset=offset)
    plt.fill(xs, ys, alpha=alpha)
    plt.plot(xs + [xs[0]], ys + [ys[0]])
    plt.gca().set_aspect("equal")


def sinusoid(amplitude, mean, phase, harmonic=1):
    """
    Float -> Float -> Float -> (Float -> Float)
    Create a sinusoid function.
    """
    return lambda t: mean + amplitude * sin(harmonic * t + phase)


def superpose(*sinusoids):
    """
    [(Float -> Float)] -> (Float -> Float)
    Superpose a number of sinusoids.
    """
    return lambda t: sum([s(t) for s in sinusoids]) / len(sinusoids)


def sample_sinusoid():
    """
    () -> (Float -> Float)
    Create a random sinusoid.
    """
    harmonic = int(uniform(1.1, 4.8))
    mean = uniform(0.1, 0.8)
    amplitude = uniform(mean * 1, mean * 3)
    phase = uniform(0.0, 2 * pi)
    return sinusoid(mean, amplitude, phase, harmonic=harmonic)


def sample_superposed_sinusoid(n):
    """
    Int -> (Float -> Float)
    Sample a number of sinusoids and superpose them.
    """
    return superpose(*[sample_sinusoid() for _ in range(n)])


def random_offset(minimum=0.0, maximum=4.0):
    """
    Float? -> Float? -> (Float, Float)
    Return a random point uniformly distributed.
    """
    return tuple(
        [
            s * (-1 if uniform(0, 1) < 0.5 else 1)
            for s in (uniform(minimum, maximum), uniform(minimum, maximum))
        ]
    )


def create_plot():
    """
    () -> ()
    Create a plot of an exemplar viable subset of S.
    """
    for _ in range(5):
        plot_coordinates(sample_superposed_sinusoid(17), offset=random_offset())
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def get_control_points():
    """
    () -> [((Float, Float), (Float, Float))]
    Generate a number of control points used for calculating the
    latent representations of the viable spaces.
    """
    quadrants = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    inners = [(uniform(0.1, x * 0.8), uniform(0.1, y * 0.8)) for x, y in quadrants]
    outers = [(uniform(12, 36) * x, uniform(12, 36) * y) for x, y in inners]
    return [(inner, outer) for inner, outer in zip(inners, outers)]


def extract_quadrilaterals(control_points):
    """
    [((Float, Float), (Float, Float))] -> [[(Float, Float)]]
    Exract the vertex coordinates of five quadrilaterals from the
    given control points.
    """
    return [
        list(a)[::-1] + list(b) for a, b in zip(control_points, cycle(control_points))
    ] + [[inner for inner, _ in control_points]]


def cycle(values):
    """
    [a] -> [a]
    Create a new list with the first value moved to the end of the list.
    """
    return values[1:] + [values[0]]
