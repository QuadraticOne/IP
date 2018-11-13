from environments.environment \
    import ContinuousEnvironment, DrawableEnvironment


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

    WIDTH = 20
    HEIGHT = 10

    def constraint_shape():
        """
        () -> [Int]
        Return the shape of a valid constraint tensor.
        """
        return [Bridge.HEIGHT, Bridge.WIDTH]

    def solution_shape():
        """
        () -> [Int]
        Return the shape of a valid solution tensor.
        """
        return Bridge.constraint_shape()

    def solve(constraint, solution):
        """
        [Float] -> [Float] -> Bool
        Test whether the parameters of the given solution satisfy the parameters
        of the constraint.
        """
        raise NotImplementedError()

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
        return Bridge.constraint_shape()

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
