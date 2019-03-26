import modules.reusablenet as rnet
import tensorflow as tf


class ExportedParametricGenerator:
    def __init__(self, generator):
        """
        ParametricGenerator -> ExportedParametricGenerator
        Export a parametric generator that has been trained for easy querying
        in typical use cases.
        """
        self.sample_for_constraint = self._make_sample_for_constraint(generator)

    def _make_sample_for_constraint(self, parametric_generator):
        """
        ParametricGenerator -> (np.array -> Int -> [Dict])
        From a generator, create a function that takes a constraint and a number
        of samples, and creates that number of output dictionaries.  Each dictionary
        contains a latent sample, solution, and satisfaction probability.
        """
        sample_size = tf.placeholder(tf.int32, shape=(), name="sample_size")
        constraint_input = rnet.make_input_node(
            [parametric_generator.constraint_dimension]
        )

        multiples = tf.stack([sample_size, tf.constant(1)])
        tiled_constraints = tf.tile(tf.expand_dims(constraint_input, 0), multiples)

        latent_samples = tf.random.uniform(
            tf.stack([sample_size, tf.constant(parametric_generator.latent_dimension)]),
            minval=parametric_generator.latent_lower_bound,
            maxval=parametric_generator.latent_upper_bound,
        )

        weights, biases = parametric_generator.build_embedder(tiled_constraints)
        generator = parametric_generator.build_generator(
            latent_samples, weights, biases
        )
        discriminator = parametric_generator.build_discriminator(
            generator["output"], tiled_constraints
        )

        def sample_for_constraint(constraint, samples):
            """
            np.array -> Int -> [Dict]
            Given a constraint and a requested number of samples, return that
            number of dictionaries, where each dictionary represents a sample from
            the latent space for that constraint.  They contain keys for 
            `latent`, `solution`, and `satisfaction_probability`.
            """
            pass

        return sample_for_constraint
