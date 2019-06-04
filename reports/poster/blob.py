from math import cos, sin, pi, exp
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
    inner_boundary = 0.3
    outer_boundary = 1.2
    quadrants = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    inners = [
        (
            uniform(x * inner_boundary, x * outer_boundary),
            uniform(y * inner_boundary, y * outer_boundary),
        )
        for x, y in quadrants
    ]
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


def get_splines():
    """
    () -> [Spline]
    Get a number of splines to be used for the latent space plot.
    """
    quadrilaterals = extract_quadrilaterals(get_control_points())
    return [
        Spline(
            [
                SplinePoint(i / len(quadrilateral), xy)
                for i, xy in enumerate(quadrilateral)
            ]
        )
        for quadrilateral in quadrilaterals
    ]


class SplinePoint:
    def __init__(self, radial_position, global_position):
        """
        Float -> (Float, Float) -> SplinePoint
        Create a single spline point.
        """
        self.radial_position = radial_position
        self.global_position = global_position
        self.x, self.y = self.global_position

    def radial_distance_to(self, t):
        """
        Float -> Float
        Calculate the radial distance to the spline point.
        """
        d = abs(t - self.radial_position)
        return min(d, 1 - d)

    def __str__(self):
        """
        () -> String
        Return a string representation of the spline point.
        """
        return "({}, {}) @ {}".format(self.x, self.y, self.radial_position)


class Spline:
    def __init__(self, spline_points):
        """
        [SplinePoint] -> Spline
        Create a spline from a series of weighted points.
        """
        self.spline_points = spline_points

    def position_at(self, t):
        """
        Float -> (Float, Float)
        Get the spline's position at the given parametric value.
        """
        weighted_positions = self.weights_at(t)
        n = len(self.spline_points)
        x_total, y_total = 0.0, 0.0
        for x, y in weighted_positions:
            x_total += x
            y_total += y
        return x_total / n, y_total / n

    def get_point_weights(self, t):
        """
        Float -> ([Float], [SplinePoint])
        Pair each point with a calculated weight.
        """
        return (
            softmax(
                [1 / (0.1 + p.radial_distance_to(t) ** 3) for p in self.spline_points]
            ),
            self.spline_points,
        )

    def weights_at(self, t):
        """
        Float -> [(Float, Float)]
        Calculate the weighted position of each spline at the given
        parametric value.
        """
        weights, points = self.get_point_weights(t)
        return [(p.x * w, p.y * w) for p, w in zip(points, weights)]

    def plot(self, segments=100, alpha=0.4):
        """
        Int? -> Float? -> ()
        Plot the spline but do not show the plot.
        """
        points = [self.position_at(t) for t in linspace(0, 1, segments)]
        xs, ys = zip(*points)
        xs = list(xs)
        ys = list(ys)

        plt.fill(xs, ys, alpha=alpha)
        plt.plot(xs + [xs[0]], ys + [ys[0]])


def softmax(xs):
    """
    [Float] -> [Float]
    Calculate the softmax of a vector of values.
    """
    exponentials = [exp(x) for x in xs]
    total = sum(exponentials)
    return [x / total for x in exponentials]


def create_latent_plot():
    """
    () -> ()
    Create a plot of the viable set in the latent space.
    """
    for spline in get_splines():
        spline.plot()
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect("equal")
    plt.show()
