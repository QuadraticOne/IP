from environments.environment import ContinuousEnvironment, DrawableEnvironment
from math import sqrt
from numpy import linspace
from random import uniform
from wise.training.samplers.anonymous import AnonymousSampler
from wise.training.samplers.feeddict import FeedDictSampler
from wise.training.samplers.resampled import BinomialResampler


class Circles(ContinuousEnvironment, DrawableEnvironment):
    """
    Defines an environment which consists of two circles: one constraint,
    and one design.  Each circle is parameterised by their x- and y-coordinates,
    mapped from the real domain to (-1, 1), and their radius, mapped from the
    real domain to (0, 1).

    The constraint is considered satisfied if it does not overlap with the
    solution at all.
    """

    # Number of pixels along each edge of the environment when
    # presented as an image
    PIXELS = 10

    def constraint_type():
        """
        () -> Class
        Return the type of the class which stores constraint data.
        """
        return type([0.0])

    def constraint_shape():
        """
        () -> [Int]
        Return the shape of a valid constraint tensor.
        """
        return [3]

    def constraint_representation(constraint):
        """
        Constraint -> Tensor
        Return a tensor representation of the given constraint.  The type
        of the constraint given must match the type returned by the
        constraint_type function.
        """
        return constraint

    @classmethod
    def constraint_dimension(cls):
        """
        () -> Int
        Return the dimensionality of this environments' constraints when
        represented as a tensor.
        """
        return 3

    @classmethod
    def flatten_constraint(cls, constraint):
        """
        Constraint -> [Float]
        Flatten the constraint from its tensor representation into a
        rank-one vector.  It is recommended that this method is overriddn
        to improve performance.
        """
        return constraint

    def solution_type():
        """
        () -> Class
        Return the type of the class which stores solution data.
        """
        return type([0.0])

    def solution_shape():
        """
        () -> [Int]
        Return the shape of a valid solution tensor.
        """
        return [3]

    def solution_representation(solution):
        """
        Solution -> Tensor
        Return a tensor representation of the given solution.  The type
        of the constraint given must match the type returned by the
        constraint_type function.
        """
        return solution

    @classmethod
    def solution_dimension(cls):
        """
        () -> Int
        Return the dimensionality of this environments' solutions when
        represented as a tensor.
        """
        return 3

    @classmethod
    def flatten_solution(cls, solution):
        """
        Solution -> [Float]
        Flatten the solution from its tensor representation into a
        rank-one vector.  It is recommended that this method is overriddn
        to improve performance.
        """
        return solution

    def image_shape(fidelity=None):
        """
        () -> [Int]
        """
        return [fidelity, fidelity]

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
        return _distance(cons_x, cons_y, sol_x, sol_y) < cons_r + sol_r

    def environment_sampler(
        constraint_input="constraint",
        solution_input="solution",
        satisfaction_input="satisfaction",
        sampler_transform=BinomialResampler.halves_on_last_element_head,
    ):
        """
        Object? -> Object? -> Object? -> (Sampler a -> Sampler a)?
            -> FeedDictSampler ([Float], [Float], [Float])
        Return a sampler that generates random constraint/solution pairs and
        matches them with the satisfaction of the constraint.
        """
        sampler = AnonymousSampler(single=Circles._make_environment)
        return FeedDictSampler(
            sampler_transform(sampler),
            {
                constraint_input: lambda t: t[0],
                solution_input: lambda t: t[1],
                satisfaction_input: lambda t: t[2],
            },
        )

    def image_shape():
        """
        () -> [Int]
        Return the shape of images output by this environment with the given
        fidelity settings.
        """
        return [Circles.PIXELS, Circles.PIXELS]

    def as_image(constraint, solution, fidelity=10):
        """
        Constraint -> Solution -> [[Float]]
        Produce an image with PIXELS^2 pixels describing the environment.
        Pixels with a value of 1 lie within one of the two circles; pixels with
        a value of 0 lie outside the circles.
        """
        cons_x, cons_y, cons_r = constraint[0], constraint[1], constraint[2]
        sol_x, sol_y, sol_r = solution[0], solution[1], solution[2]
        steps = linspace(-1, 1, Circles.PIXELS)
        reversed_steps = steps[::-1]

        def inside_circle(x, y):
            return (
                _distance(x, y, cons_x, cons_y) < cons_r
                or _distance(x, y, sol_x, sol_y) < sol_r
            )

        return [
            [1.0 if inside_circle(x, y) else 0.0 for x in steps] for y in reversed_steps
        ]

    def pixel_environment_sampler(
        pixels_input="pixels",
        satisfaction_input="satisfaction",
        sampler_transform=BinomialResampler.halves_on_last_element_head,
    ):
        """
        Object? -> Object? -> (Sampler ([[Float]], [Float]) -> Sampler a)?
            -> FeedDictSampler a
        Return a sampler that generates pixel representations of environments and
        puts them into a feed dict along with their satisfactions.
        """

        def generate_pixels():
            cons, sol, satisfied = Circles._make_environment()
            pixels = Circles.as_image(cons, sol, Circles.PIXELS)
            return pixels, satisfied

        sampler = AnonymousSampler(single=generate_pixels)
        return FeedDictSampler(
            sampler_transform(sampler),
            {pixels_input: lambda t: t[0], satisfaction_input: lambda t: t[1]},
        )

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
