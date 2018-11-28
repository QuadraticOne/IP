from environments.environment \
    import ContinuousEnvironment, DrawableEnvironment
from maths.activations import identity
from random import uniform


class Bridge(ContinuousEnvironment, DrawableEnvironment):
    """
    An environment consisting of a bridge, drawn in pixels in a 2D plane,
    over an obstacle.  The bridge is defined by having each pixel turned on
    to varying intensities: darker pixels (values closer to 1) have higher
    intensity - and therefore higher strength - but also greater mass.  The
    constraint is defined in terms of a pixel matrix of the same size,
    each pixel in the constraint denoting the maximum intensity of the same
    pixel in the solution.

    Note that the pixels are stored as a list of rows.
    """

    WIDTH = 3
    HEIGHT = 2

    # Number of members to either side of a block which are able to support
    # it, not including the one directly beneath it
    SUPPORTING_MEMBERS = 1

    SELF_LOAD_FACTOR = 0.25
    LOAD_PROPAGATION_FACTOR = 1.0

    def constraint_type():
        """
        () -> Class
        Return the type of the class which stores constraint data.
        """
        return type([[[0.]]])

    def constraint_shape():
        """
        () -> [Int]
        Return the shape of a valid constraint tensor.
        """
        return [Bridge.HEIGHT, Bridge.WIDTH, 2]

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
        return Bridge.HEIGHT * Bridge.WIDTH * 2

    @classmethod
    def flatten_constraint(cls, constraint):
        """
        Constraint -> [Float]
        Flatten the constraint from its tensor representation into a
        rank-one vector.  It is recommended that this method is overriddn
        to improve performance.
        """
        return _flatten_tensor(cls.constraint_representation(constraint))

    def solution_type():
        """
        () -> Class
        Return the type of the class which stores solution data.
        """
        return type([[0.]])

    def solution_shape():
        """
        () -> [Int]
        Return the shape of a valid solution tensor.
        """
        return [Bridge.HEIGHT, Bridge.WIDTH]

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
        return Bridge.HEIGHT * Bridge.WIDTH

    @classmethod
    def flatten_solution(cls, solution):
        """
        Solution -> [Float]
        Flatten the solution from its tensor representation into a
        rank-one vector.  It is recommended that this method is overriddn
        to improve performance.
        """
        output = []
        for row in solution:
            output += row
        return output

    def solve(constraint, solution):
        """
        [Float] -> [Float] -> Bool
        Test whether the parameters of the given solution satisfy the parameters
        of the constraint.
        """
        return Bridge._is_structurally_stable(solution) and \
            Bridge._within_allowed_ranges(constraint, solution)

    def _is_structurally_stable(solution):
        """
        [[Float]] -> Bool
        Determines whether or not the given solution can hold its own weight
        according to some set of rules.
        """
        return Bridge._elementwise_predicate(lambda s, l: s < l, solution,
            Bridge._create_load_map(solution))

    def _create_load_map(solution):
        """
        [[Float]] -> [[Float]]
        Calculate the load of each block.
        """
        loads = [[cell * Bridge.SELF_LOAD_FACTOR for cell in row] for row in solution]

        for row in range(Bridge.HEIGHT - 1):
            for column in range(Bridge.WIDTH):
                supporting_indices, supporting_weights = \
                    Bridge._get_supporting_indices_and_strengths(solution, row, column)
                propagated_loads = Bridge._calculate_propagated_loads(
                    solution[row][column], supporting_weights)
                for index, additional_load in zip(supporting_indices, propagated_loads):
                    loads[row + 1][index] += additional_load
        
        return loads
                
    def _get_supporting_indices_and_strengths(solution, row, column):
        """
        [[Float]] -> Int -> Int -> Range -> [Float]
        Given a solution and a row and column specifying a block in the solution,
        return a range of column indices and a list of their corresponding weights.
        """
        columns = range(max([0, column - Bridge.SUPPORTING_MEMBERS]),
            min([column + Bridge.SUPPORTING_MEMBERS + 1, Bridge.WIDTH]))
        weights = [solution[row + 1][c] for c in columns]
        return columns, weights

    def _calculate_propagated_loads(source, targets):
        """
        Float -> [Float] -> [Float]
        Calculate the amount of load that is propagated from the source block to the
        target blocks, assuming that the amount of load transferred is equal to the
        strength of the source block multiplied by the global load propagation
        factor.
        """
        propagated_load = source * Bridge.LOAD_PROPAGATION_FACTOR
        total_distribution_weight = sum(targets)
        if total_distribution_weight == 0.0:
            even_share = propagated_load / len(targets)
            return [even_share] * len(targets)
        return [propagated_load * (weight / total_distribution_weight) \
            for weight in targets]

    def _within_allowed_ranges(constraint, solution):
        """
        [[[Float]]] -> [[Float]] -> Bool
        Determine whether or not every value in the solution is within its designated
        range as specified by the constraint.
        """
        return Bridge._elementwise_predicate(lambda c, s: c[0] <= s <= c[1])

    def _elementwise_predicate(predicate, solution, constraint):
        """
        (Float -> Float -> Bool) -> [[Float]] -> [[Float]] -> Bool
        Determine whether or not the given solution and constraint satisfy the
        predicate in all blocks, where the predicate is applied one by one to
        each pair of corresponding blocks.
        """
        for i in range(Bride.HEIGHT):
            for j in range(Bridge.WIDTH):
                if not predicate(solution[i][j], constraint[i][j]):
                    return False
        return True

    def environment_sampler(constraint_input='constraint', solution_input='solution',
            satisfaction_input='satisfaction', sampler_transform=identity):
        """
        Object? -> Object? -> Object? -> (Sampler a -> Sampler a)?
            -> FeedDictSampler ([Float], [Float], [Float])
        Return a sampler that generates random constraint/solution pairs and
        matches them with the satisfaction of the constraint.  The raw sampler is
        mapped through a user-provided transform, optionally producing a mapped
        sampler, before being extracted into a FeedDictSampler.
        """
        raise NotImplementedError()

    def image_shape(fidelity=None):
        """
        () -> [Int]
        Return the shape of images output by this environment with the given
        fidelity settings.
        """
        return [2 * Bridge.HEIGHT, Bridge.WIDTH]

    def as_image(constraint, solution, fidelity=None):
        """
        [Float] -> [Float] -> Object? -> [[Float]]
        Produce a greyscale image of the environment from the given environment
        parameters, and optionally some measure of fidelity.  Pixel values
        should be in the interval [0, 1], where 0 represents fully off and 1
        represents fully on.
        """
        output = []
        for row in solution:
            output.append(row)
        for row in constraint:
            output.append([cell[0] for cell in row])
        for row in constraint:
            output.append([cell[1] for cell in row])
        return output

    def pixel_environment_sampler(pixels_input='pixels', satisfaction_input='satisfaction',
            fidelity=None, sampler_transform=identity):
        """
        Object? -> Object? -> Object? -> (Sampler a -> Sampler a)?
            -> FeedDictSampler ([[Float]], [Float])
        Sample from the space of environments, returning them as the output of
        a FeedDictSampler in pixel format, grouped with their satisfaction.
        The raw sampler is mapped through a user-provided transform, optionally producing
        a mapped sampler, before being extracted into a FeedDictSampler.
        """
        raise NotImplementedError()


class BridgeFactory:
    """
    A collection of functions for creating training data for the
    Bridge environment.
    """

    @staticmethod
    def blank_constraint():
        """
        () -> [[[Float]]]
        Return an empty constraint (one where the allowable range of each pixel
        is [0, 1]).
        """
        return [[[0., 1.] for _ in range(Bridge.WIDTH)] for _ in range(Bridge.HEIGHT)]

    @staticmethod
    def set_lower_bound(constraint, lower_bound_on_index):
        """
        [[[Float]]] -> (Int -> Int -> Float?) -> [[[Float]]]
        Set the lower bound of each pixel to a function of its index.  If
        the function returns None, the lower bound of that pixel will not
        be altered.  Rows are 0-indexed from top to bottom, and columns
        are 0-indexed from left to right, in a list-of-rows configuration.
        """
        for row in range(Bridge.HEIGHT):
            for column in range(Bridge.WIDTH):
                x = lower_bound_on_index(row, column)
                if x is not None:
                    constraint[row][column][0] = x

    @staticmethod
    def set_upper_bound(constraint, upper_bound_on_index):
        """
        [[[Float]]] -> (Int -> Int -> Float?) -> [[[Float]]]
        Set the upper bound of each pixel to a function of its index.  If
        the function returns None, the upper bound of that pixel will not
        be altered.  Rows are 0-indexed from top to bottom, and columns
        are 0-indexed from left to right, in a list-of-rows configuration.
        """
        for row in range(Bridge.HEIGHT):
            for column in range(Bridge.WIDTH):
                x = upper_bound_on_index(row, column)
                if x is not None:
                    constraint[row][column][1] = x

    @staticmethod
    def solution_from_indices(cell_value_on_index):
        """
        (Int -> Int -> Float) -> [[Float]]
        Return a solution whose pixels are a function of their index.
        Rows are 0-indexed from top to bottom, and columns are 0-indexed from
        left to right, in a list-of-rows configuration.
        """
        return [[cell_value_on_index(row, column) \
            for column in range(Bridge.WIDTH)] for row in range(Bridge.HEIGHT)]

    @staticmethod
    def blank_solution():
        """
        () -> [[Float]]
        Create a solution whose cells are all set to 0.
        """
        return BridgeFactory.solution_from_indices(lambda _1, _2: 0.)

    @staticmethod
    def uniform_solution():
        """
        () -> [[Float]]
        Create a solution whose cells' strengths are sampled from a uniform
        distribution.
        """
        return BridgeFactory.solution_from_indices(lambda _1, _2: uniform(0, 1))
