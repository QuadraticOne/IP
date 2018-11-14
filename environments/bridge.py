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

    LOAD_PROPAGATION_FACTOR = 1.0

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

    def _is_structurally_stable(solution):
        """
        [[Float]] -> Bool
        Determines whether or not the given solution can hold its own weight
        according to some set of rules.
        """
        pass

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

    def _avoids_disallowed_areas(solution, constraint):
        """
        [[Float]] -> [[Float]] -> Bool
        Determine whether or not the given solution avoids areas that are disallowed
        by the constraint.  The constraint consists of a limit on the strength of the
        block in the corresponding slot in the solution: the strength of the block
        must be less than or equal to its slot in the constraint.
        """
        for i in range(Bride.HEIGHT):
            for j in range(Bridge.WIDTH):
                if solution[i][j] > constraint[i][j]:
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
