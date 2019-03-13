from wise.util.tensors import placeholder_node
import tensorflow as tf


class ParametricGenerator:

    def __init__(self, solution_dimension, latent_dimension,
            constraint_dimension):
        """
        Int -> Int -> Int -> ParametricGenerator
        Create a parametric generator by specifying the dimensions
        of all relevant spaces.
        """
        self.solution_dimension = solution_dimension
        self.latent_dimension = latent_dimension
        self.constraint_dimension = constraint_dimension

        self.generator_training_batch_size = 64

    def build_input_nodes(self):
        """
        () -> ()
        Construct input placeholder nodes for the tensorflow graph.
        """
        self.solution_input = placeholder_node('solution_input',
            [self.solution_dimension], dynamic_dimensions=1)
        self.latent_input = placeholder_node('latent_input',
            [self.latent_dimension], dynamic_dimensions=1)
        self.constraint_input = placeholder_node('constraint_input',
            [self.constraint_dimension], dynamic_dimensions=1)

    def build_sample_nodes(self):
        """
        () -> ()
        Construct sample nodes for training the generator.
        """
        self.latent_sample = tf.random.uniform(
            [self.generator_training_batch_size, self.latent_dimension])
