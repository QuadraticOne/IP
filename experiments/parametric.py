from modules.generator import ParametricGenerator
import modules.reusablenet as rnet
import tensorflow as tf


class ParametricGeneratorTest(ParametricGenerator):
    def __init__(self, latent_dimension, constraint_dimension, embedding_dimension):
        """
        Int -> Int -> Int -> ParametricGeneratorTest
        Create an object designed to test the parametric generator training
        algorithm on a one-dimensional environment.
        """
        super().__init__(
            "one_dim_test",
            1,
            latent_dimension,
            constraint_dimension,
            embedding_dimension,
        )

    def make_constraint_sample_node(self):
        """
        () -> tf.Node
        Create an instance of a node sampling from the constraint space in
        a manner likely to create a realistic distribution.
        """
        return tf.constant([[1.0, 0.0, -1.0]] * self.generator_training_batch_size)

    def build_discriminator(self, solution_input, constraint_input):
        """
        tf.Node -> tf.Node -> Dict
        Build the discriminator, given nodes containing batches of
        solutions and constraints respectively.
        """
        tiled_x = tf.tile(solution_input, [1, self.constraint_dimension])
        return tf.reduce_mean(
            self.sigmoid_bump(tiled_x, constraint_input),
            name=self.extend_name("objective_function"),
        )

    def sigmoid_bump(self, x, offset, width=5.0, fatness=0.05, y_scale=1.0):
        """
        tf.Node -> tf.Node -> Float? -> Float? -> Float? -> tf.Node
        Create a Tensorflow graph representing a sigmoid bump.
        """
        after_offset = (x - offset) / fatness
        return y_scale * (
            tf.sigmoid(after_offset + width) - tf.sigmoid(after_offset - width)
        )


def run():
    """
    () -> ()
    Attempt to train a parametric generate to sample from the solution space
    of an objective function with a constraint argument.
    """
    r = "leaky-relu"
    l = (10, "leaky-relu")

    p = ParametricGeneratorTest(5, 3, 10)
    p.set_embedder_architecture([l], r, [l], r)
    p.set_generator_architecture([l, l], r, "tanh")

    p.build_input_nodes()
    p.build_sample_nodes()

    w, b = p.build_embedder(p.constraint_sample)
    g = p.build_generator(p.latent_sample, w, b)

    s = tf.Session()
    s.run(tf.global_variables_initializer())

    f = p.build_discriminator(g["output"], p.constraint_sample)

    print(s.run([g["output"][0], f]))
