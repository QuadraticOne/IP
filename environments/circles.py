from environments.environment import ContinuousEnvironment
from math import sqrt
from numpy import linspace
from random import uniform
from wise.training.samplers.anonymous import AnonymousSampler
from wise.training.samplers.feeddict import FeedDictSampler


class Circles:
    """
    Defines an environment which consists of two circles: one constraint,
    and one design.  Each circle is parameterised by their x- and y-coordinates,
    mapped from the real domain to (-1, 1), and their radius, mapped from the
    real domain to (0, 1).

    The constraint is considered satisfied if it does not overlap with the
    solution at all.
    """

    def solve(constraint, solution):
        """
        [Float] -> [Float] -> Bool
        Test whether the parameters of the given solution satisfy the parameters
        of the constraint.  Both lists are expected to contain 3 floats, each
        specifying the x- and y-coordinates of the circle's origin and its radius,
        after normalisation.
        """
        cons_x, cons_y, cons_r = constraint[0], constraint[1], constraint[2]
        sol_x, sol_y, sol_r = solution[0], solution[1], solution[2]
        return _distance(cons_x, cons_y, sol_x, sol_y) > cons_r + sol_r

    def as_image(constraint, solution, fidelity=10):
        """
        [Float] -> [Float] -> Int? -> [[Float]]
        Produce an image with fidelity^2 pixels describing the environment.
        Pixels with a value of 1 lie within one of the two circles; pixels with
        a value of 0 lie outside the circles.
        """
        cons_x, cons_y, cons_r = constraint[0], constraint[1], constraint[2]
        sol_x, sol_y, sol_r = solution[0], solution[1], solution[2]
        steps = linspace(-1, 1, fidelity + 1)
        reversed_steps = steps[::-1]
        def inside_circle(x, y):
            return _distance(x, y, cons_x, cons_y) < cons_r \
                or _distance(x, y, sol_x, sol_y) < sol_r
        return [
            [1. if inside_circle(x, y) else 0. for x in steps] \
                for y in reversed_steps
        ]

    def environment_sampler(constraint_input='constraint', solution_input='solution',
            satisfaction_input='satisfaction'):
        """
        Object? -> Object? -> Object? -> FeedDictSampler ([Float], [Float], [Float])
        Return a sampler that generates random constraint/solution pairs and
        matches them with the satisfaction of the constraint.
        """
        sampler = AnonymousSampler(single=Circles._make_environment)
        return FeedDictSampler(sampler, {
            constraint_input: lambda t: t[0],
            solution_input: lambda t: t[1],
            satisfaction_input: lambda t: t[2]
        })

    def pixel_environment_sampler(pixels_input='pixels',
            satisfaction_input='satisfaction', fidelity=10):
        """
        Object? -> Object? -> Int? -> FeedDictSampler ([[Float]], [Float])
        Return a sampler that generates pixel representations of environments and
        puts them into a feed dict along with their satisfactions.
        """
        def generate_pixels():
            cons, sol, satisfied = Circles._make_environment()
            pixels = Circles.as_image(cons, sol, fidelity)
            return pixels, satisfied
        sampler = AnonymousSampler(single=generate_pixels)
        return FeedDictSampler(sampler, {
            pixels_input: lambda t: t[0],
            satisfaction_input: lambda t: t[1]
        })

    def _make_circle():
        return [uniform(-1, 1), uniform(-1, 1), uniform(0, 1)]

    def _make_environment():
        cons = Circles._make_circle()
        sol = Circles._make_circle()
        satisfied = [1] if Circles.solve(cons, sol) else [0]
        return cons, sol, satisfied


def _distance(x0, y0, x1, y1):
    """
    Float -> Float -> Float -> Float -> Float
    Determine the Euclidian distance between two points.
    """
    def sqr(x):
        return x * x

    return sqrt(sqr(x1 - x0) + sqr(y0 - y1))
