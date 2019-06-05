from maths.activations import identity
from wise.training.samplers.mapped import MappedSampler
from wise.training.samplers.anonymous import AnonymousSampler
from wise.training.samplers.resampled import BinomialResampler
from wise.training.samplers.dataset import DataSetSampler
import matplotlib.pyplot as plt
import numpy as np


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

    @classmethod
    def constraint_dimension(cls):
        """
        () -> Int
        Return the dimensionality of this environments' constraints when
        represented as a tensor.
        """
        return _product(cls.constraint_shape())

    @classmethod
    def flatten_constraint(cls, constraint):
        """
        Constraint -> [Float]
        Flatten the constraint from its tensor representation into a
        rank-one vector.  It is recommended that this method is overridden
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

    @classmethod
    def solution_dimension(cls):
        """
        () -> Int
        Return the dimensionality of this environments' solutions when
        represented as a tensor.
        """
        return _product(cls.solution_shape())

    @classmethod
    def flatten_solution(cls, solution):
        """
        Solution -> [Float]
        Flatten the solution from its tensor representation into a
        rank-one vector.  It is recommended that this method is overridden
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

    def environment_sampler(
        constraint_input="constraint",
        solution_input="solution",
        satisfaction_input="satisfaction",
        sampler_transform=identity,
    ):
        """
        Object? -> Object? -> Object? -> (Sampler (Constraint, Solution, Bool)
            -> Sampler a)? -> FeedDictSampler a
        Return a sampler that generates random constraint/solution pairs and
        matches them with the satisfaction of the constraint.  The raw sampler is
        mapped through a user-provided transform, optionally producing a mapped
        sampler, before being extracted into a FeedDictSampler.
        """
        raise NotImplementedError()

    @classmethod
    def environment_representation_sampler(
        cls,
        constraint_input="constraint",
        solution_input="solution",
        satisfaction_input="satisfaction",
        sampler_transform=identity,
    ):
        """
        Object? -> Object? -> Object? -> (Sampler (Tensor, Tensor, Bool) ->
            Sampler a)? -> FeedDictSampler a
        Return a sampler that generates random constraint/solution pairs and
        matches their tensor representations with the satisfaction of the constraint.
        The raw sampler is mapped through a user-provided transform, optionally
        producing a mapped sampler, before being extracted into a FeedDictSampler.
        """

        def take_reps(constraint_solution_satisfaction):
            con, sol, sat = constraint_solution_satisfaction
            return (
                cls.constraint_representation(con),
                cls.solution_representation(sol),
                sat,
            )

        return cls.environment_sampler(
            constraint_input,
            solution_input,
            satisfaction_input,
            lambda s: sampler_transform(MappedSampler(s, take_reps)),
        )

    @classmethod
    def flattened_environment_representation_sampler(
        cls,
        constraint_input="constraint",
        solution_input="solution",
        satisfaction_input="satisfaction",
        sampler_transform=identity,
    ):
        """
        Object? -> Object? -> Object? -> (Sampler ([Float], [Float], Bool)
            -> Sampler a)? -> FeedDictSampler a
        Return a sampler that generates random constraint/solution pairs and
        matches their flattened representations with the satisfaction of the
        constraint.  The raw sampler is mapped through a user-provided transform,
        optionally producing a mapped sampler, before being extracted into a
        FeedDictSampler.
        """

        def take_reps(constraint_solution_satisfaction):
            con, sol, sat = constraint_solution_satisfaction
            return cls.flatten_constraint(con), cls.flatten_solution(sol), sat

        return cls.environment_sampler(
            constraint_input,
            solution_input,
            satisfaction_input,
            lambda s: sampler_transform(MappedSampler(s, take_reps)),
        )

    @staticmethod
    def draw(image):
        """
        [[Float]] -> ()
        Plot the given matrix as a greyscale image.
        """
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.show()


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
        return type([[0.0]])

    def image_shape():
        """
        () -> [Int]
        Return the shape of images output by this environment with the given
        fidelity settings.
        """
        raise NotImplementedError()

    @classmethod
    def image_dimension(cls):
        """
        () -> Int
        Return the number of pixels in the image.
        """
        return _product(cls.image_shape)

    def as_image(constraint, solution):
        """
        Constraint -> Solution -> [[Float]]
        Produce a greyscale image of the environment from the given environment
        parameters.  Pixel values should be in the interval [0, 1], where 0
        represents fully off and 1 represents fully on.
        """
        raise NotImplementedError()

    @classmethod
    def as_flattened_image(cls, constraint, solution):
        """
        Constraint -> Solution -> [Float]
        Convert the constraint and solution into an image, then flatten it
        into a rank-one tensor.
        """
        image = cls.as_image(constraint, solution)
        output = []
        for row in image:
            output += row
        return output

    def pixel_environment_sampler(
        pixels_input="pixels",
        satisfaction_input="satisfaction",
        sampler_transform=identity,
    ):
        """
        Object? -> Object? -> (Sampler ([[Float]], [Float]) -> Sampler a)?
            -> FeedDictSampler a
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


class VectorEnvironment:
    def sample_solution(self):
        """
        () -> np.array
        Return a numpy array representing a possible solution.
        """
        raise NotImplementedError()

    def sample_constraint(self):
        """
        () -> np.array
        Return a numpy array representing a possible constraint.
        """
        raise NotImplementedError()

    def satisfaction(self, solution_constraint_pair):
        """
        (np.array -> np.array) -> Bool
        Determine whether a solution satisfies the constraint.
        """
        raise NotImplementedError()

    def sample_solution_constraint_pair(self):
        """
        () -> (np.array, np.array)
        Take a solution and constraint and merge them into a tuple.
        By default these samples are independent.
        """
        return self.sample_solution(), self.sample_constraint()

    def sample_training_tuple(self):
        """
        () -> (np.array, np.array, Bool)
        Sample a solution, constraint, satisfaction tuple that can be used
        for training.
        """
        solution, constraint = self.sample_solution_constraint_pair()
        return solution, constraint, self.satisfaction((solution, constraint))

    def sampler(self, balanced=True):
        """
        Bool? -> Sampler (np.array, np.array, Float)
        Return a sampler for solutions, constraints, and satisfactions from
        the environment.  This sampler can optionally be balanced to produce
        positive and negative samples with equal probability.
        """
        sampler = AnonymousSampler(single=self.sample_training_tuple)
        return (
            BinomialResampler.halves_on_last_element(sampler) if balanced else sampler
        )

    def dataset_sampler(self, size, balanced=True):
        """
        Int -> Bool? -> DatasetSampler (np.array, np.array, Bool)
        Return a sampler which samples only from a fixed subset of samples of
        the environment.
        """
        return DataSetSampler.from_sampler(self.sampler(balanced=balanced), size)

    def make_dataset(self, size, balanced=True):
        """
        Int -> Bool? -> [(np.array, np.array, Bool)]
        Create a dataset of environment samples.
        """
        return self.dataset_sampler(size, balanced=balanced).points

    def save_dataset(self, size, directory, file_name, balanced=True):
        """
        Int -> String -> String -> Bool? -> ()
        Save a dataset of environment samples to the given location.
        """
        self.dataset_sampler(size, balanced=balanced).save(directory, file_name)

    @staticmethod
    def load_dataset(directory, file_name, return_sampler=False):
        """
        String -> String -> Bool? -> [(np.array, np.array, Bool)]
        Load a pre-saved dataset of samples from an environment.
        """
        sampler = DataSetSampler.restore(directory, file_name)
        return sampler if return_sampler else sampler.points


class UniformVectorEnvironment(VectorEnvironment):
    def __init__(
        self,
        solution_dimension,
        constraint_dimension,
        satisfaction,
        solution_range=(0.0, 1.0),
        constraint_range=(0.0, 1.0),
    ):
        """
        Int -> Int -> -> ((np.array, np.array) -> Bool) ->(Float, Float)?
            -> (Float, Float)? -> UniformVectorEnvironment
        Create a uniform vector environment, whose solutions and constraints
        are, by default, sampled from a uniform space in m or n dimensions
        respectively.  The objective function can be provided as a lambda.
        """
        self.solution_dimension = solution_dimension
        self.constraint_dimension = constraint_dimension
        self.solution_range = solution_range
        self.constraint_range = constraint_range

        self._satisfaction = satisfaction

    def sample_solution(self):
        """
        () -> np.array
        Return a numpy array representing a possible solution.
        """
        return np.float32(
            np.random.uniform(
                low=self.solution_range[0],
                high=self.solution_range[1],
                size=[self.solution_dimension],
            )
        )

    def sample_constraint(self):
        """
        () -> np.array
        Return a numpy array representing a possible constraint.
        """
        return np.float32(
            np.random.uniform(
                low=self.constraint_range[0],
                high=self.constraint_range[1],
                size=[self.constraint_dimension],
            )
        )

    def satisfaction(self, solution_constraint_pair):
        """
        (np.array, np.array) -> Bool
        Determine whether a solution satisfies the constraint.
        """
        return self._satisfaction(solution_constraint_pair)
