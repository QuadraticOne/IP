from maths.activations import identity


class ContinuousEnvironment:
    """
    Base class for an environment in which the solution space and
    constraint space are both continuous.
    """

    def constraint_type():
        """
        () -> Class
        Return the type of the class which stores constraint data.
        """
        raise NotImplementedError()

    def constraint_shape():
        """
        () -> [Int]
        Return the shape of a valid constraint tensor.
        """
        raise NotImplementedError()

    def constraint_representation(constraint):
        """
        Constraint -> Tensor
        Return a tensor representation of the given constraint.  The type
        of the constraint given must match the type returned by the
        constraint_type function.
        """
        raise NotImplementedError()

    def constraint_dimension(cls):
        """
        () -> Int
        Return the dimensionality of this environments' constraints when
        represented as a tensor.
        """
        return _product(cls.constraint_shape())

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
        raise NotImplementedError()

    def solution_shape():
        """
        () -> [Int]
        Return the shape of a valid solution tensor.
        """
        raise NotImplementedError()

    def solution_representation(solution):
        """
        Solution -> Tensor
        Return a tensor representation of the given solution.  The type
        of the constraint given must match the type returned by the
        constraint_type function.
        """
        raise NotImplementedError()

    def solution_dimension(cls):
        """
        () -> Int
        Return the dimensionality of this environments' solutions when
        represented as a tensor.
        """
        return _product(cls.solution_shape())

    def flatten_solution(cls, solution):
        """
        Solution -> [Float]
        Flatten the solution from its tensor representation into a
        rank-one vector.  It is recommended that this method is overriddn
        to improve performance.
        """
        return _flatten_tensor(cls.solution_representation(solution))

    def solve(constraint, solution):
        """
        Constraint -> Solution -> Bool
        Test whether the parameters of the given solution satisfy the parameters
        of the constraint.
        """
        raise NotImplementedError()

    def environment_sampler(constraint_input='constraint', solution_input='solution',
            satisfaction_input='satisfaction', sampler_transform=identity):
        """
        Object? -> Object? -> Object? -> (Sampler (Constraint, Solution, Bool) -> Sampler a)?
            -> FeedDictSampler (Tensor, Tensor, [Float])
        Return a sampler that generates random constraint/solution pairs and
        matches them with the satisfaction of the constraint.  The raw sampler is
        mapped through a user-provided transform, optionally producing a mapped
        sampler, before being extracted into a FeedDictSampler.
        """
        raise NotImplementedError()


class DrawableEnvironment:
    """
    Base class for an environment in which the solution space and
    constraint space can both be represented by a 2D greyscale image.
    """

    def image_type():
        """
        () -> Class
        Return the type of the class which stores image data.  By default, this
        is a list of lists of floats.
        """
        return type([[0.]])

    def image_shape(fidelity=None):
        """
        () -> [Int]
        Return the shape of images output by this environment with the given
        fidelity settings.
        """
        raise NotImplementedError()

    def as_image(constraint, solution, fidelity=None):
        """
        [Float] -> [Float] -> Object? -> [[Float]]
        Produce a greyscale image of the environment from the given environment
        parameters, and optionally some measure of fidelity.  Pixel values
        should be in the interval [0, 1], where 0 represents fully off and 1
        represents fully on.
        """
        raise NotImplementedError()

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


def _product(values):
    """
    [Int] -> Int
    Return the product of a list of values.
    """
    current_value = 1
    for value in values:
        current_value *= value
    return current_value


def _flatten_tensor(tensor):
    """
    Tensor -> [Float]
    Flatten the given tensor into a rank-one list of floats.
    """
    if _is_list(tensor):
        output = []
        for subtensor in tensor:
            output += _flatten_tensor(subtensor)
        return output
    else:
        return [tensor]


def _is_list(value):
    """
    Object -> Bool
    Determine whether or not the value is a list.
    """
    try:
        _ = value[0]
        return True
    except:
        return False
