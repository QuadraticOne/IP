from environments.environment \
    import ContinuousEnvironment, DrawableEnvironment
from maths.activations import identity
from random import uniform
from numpy import reshape, vectorize
from scipy.optimize import minimize
from math import exp


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

    WIDTH = 6
    HEIGHT = 4

    # Number of members to either side of a block which are able to support
    # it, not including the one directly beneath it
    SUPPORTING_MEMBERS = 3

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
        return reshape(constraint, (cls.constraint_dimension()))

    @classmethod
    def unflatten_constraint(cls, flattened_constraint):
        """
        [Float] -> Constraint
        Unflatten the solution from its representation as a rank-one
        vector to its normal data type.
        """
        return reshape(flattened_constraint, cls.constraint_shape())

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
        return reshape(solution, cls.solution_dimension())

    @classmethod
    def unflatten_solution(cls, flattened_solution):
        """
        [Float] -> Solution
        Unflatten the solution from its representation as a rank-one
        vector to its normal data type.
        """
        return reshape(flattened_solution, cls.solution_shape())

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
        return Bridge._elementwise_predicate(lambda c, s: c[0] <= s <= c[1],
            constraint, solution)

    def _elementwise_predicate(predicate, solution, constraint):
        """
        (Float -> Float -> Bool) -> [[Float]] -> [[Float]] -> Bool
        Determine whether or not the given solution and constraint satisfy the
        predicate in all blocks, where the predicate is applied one by one to
        each pair of corresponding blocks.
        """
        for i in range(Bridge.HEIGHT):
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
        _solution = solution if type(solution) == type([[0.]]) else solution.tolist()
        output = []
        for row in _solution:
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

    EPS = 1e-3

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

    @staticmethod
    def set_inclusion_zone(constraint, is_in_zone, threshold):
        """
        [[[Float]]] -> (Int -> Int -> Bool) -> Float -> ()
        Alter a constraint such that the minimum value of each cell in the
        inclusion zone is the given threshold, and the maximum value is 1.
        Whether or not a cell is in the inclusion zone is determined by passing
        its coordinates (from the top left) to the `is_in_zone` lambda.
        """
        y = 0
        for row in constraint:
            x = 0
            for column in row:
                if is_in_zone(x, y):
                    constraint[y][x] = [threshold, 1.]
                x += 1
            y += 1
        
    @staticmethod
    def set_exclusion_zone(constraint, is_in_zone, threshold=EPS):
        """
        [[[Float]]] -> (Int -> Int -> Bool) -> Float? -> ()
        Alter a constraint such that the maximum value of each cell in the
        inclusion zone is the given threshold, and the minimum value is 0.
        Whether or not a cell is in the exclusion zone is determined by passing
        its coordinates (from the top left) to the `is_in_zone` lambda.
        """
        y = 0
        for row in constraint:
            x = 0
            for column in row:
                if is_in_zone(x, y):
                    constraint[y][x] = [0., threshold]
                x += 1
            y += 1

    @staticmethod
    def map_to_allowable_range(constraint, solution):
        """
        [[[Float]]] -> [[Float]] -> ()
        Alter the values of the solution such that they fall in the
        allowable range of their cell by the same proportion that they
        were previously between 0 and 1.
        """
        for row in range(Bridge.HEIGHT):
            for column in range(Bridge.WIDTH):
                min_value = constraint[row][column][0]
                max_value = constraint[row][column][1]
                solution[row][column] = min_value + \
                    (max_value - min_value) * solution[row][column]

    @staticmethod
    def objective_function(constraint):
        """ 
        [[[Float]]] -> ([[Float]] -> Float)
        Create an objective function on the solution space for the given
        constraint.  The objective function utilises the inner workings of
        the bridge by minimising the stress of the most overstressed cell,
        and so should only be used for dataset creation.
        """
        def maximum_stress(solution):
            unflattened_solution = BridgeFactory.preprocess_solution(constraint,
                solution)
            load_map = Bridge._create_load_map(unflattened_solution)

            current_max_overstress = 0.
            for row in range(Bridge.HEIGHT):
                for column in range(Bridge.WIDTH):
                    cell_overstress = load_map[row][column] - \
                        unflattened_solution[row][column]
                    if cell_overstress > current_max_overstress:
                        current_max_overstress = cell_overstress
            return current_max_overstress

        return maximum_stress

    @staticmethod
    def preprocess_solution(constraint, raw_solution):
        """
        [[[Float]]] -> [Float] -> [[Float]]
        Take a solution whose cells may have any real value, squish them to
        be between 1 and 0, and then map these to the allowable range of the
        solution.  Then return the result.  The solution must be in the form
        of a rank-one numpy ndarray.
        """
        sigmoid = vectorize(lambda x: 1. / (1 + exp(-x)))
        solution = Bridge.unflatten_solution(sigmoid(raw_solution))
        BridgeFactory.map_to_allowable_range(constraint, solution)
        return solution

    @staticmethod
    def find_viable_design(constraint):
        """
        [[[Float]]] -> (Float, Bool, [[Float]])
        Attempt to find a bridge design which satisfies the constraints.
        Return the design along with the amount by which the most stressed
        cell is overstressed and whether or not the optimisation terminated
        fully.
        """
        objective_function = BridgeFactory.objective_function(constraint)
        ansatz = BridgeFactory.uniform_solution()
        result = minimize(objective_function, ansatz)
        return result.fun, result.success, \
            BridgeFactory.preprocess_solution(constraint, result.x)
