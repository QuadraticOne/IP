from modules.parametric.generator import ParametricGenerator
from modules.parametric.training import optimiser
from wise.training.routines import fit
import modules.reusablenet as rnet
import tensorflow as tf


class ParametricGeneratorTest(ParametricGenerator):
    def __init__(self, constraint_dimension, embedding_dimension, proxy_weight):
        """
        Int -> Int -> Float -> ParametricGeneratorTest
        Create an object designed to test the parametric generator training
        algorithm on a one-dimensional environment.
        """
        super().__init__(
            "one_dim_test", 1, 1, constraint_dimension, embedding_dimension
        )

        self.proxy_weight = proxy_weight

    def make_constraint_sample_node(self):
        """
        () -> tf.Node
        Create an instance of a node sampling from the constraint space in
        a manner likely to create a realistic distribution.
        """
        return tf.constant([[-0.8, 0.0, 0.8]] * self.generator_training_batch_size)

    def build_discriminator(self, solution_input, constraint_input):
        """
        tf.Node -> tf.Node -> Dict
        Build the discriminator, given nodes containing batches of
        solutions and constraints respectively.
        """
        tiled_x = tf.tile(solution_input, [1, self.constraint_dimension])
        summed_output = tf.reduce_mean(
            self.sigmoid_bump(tiled_x, constraint_input), axis=1
        )
        return {
            "solution_input": solution_input,
            "constraint_input": constraint_input,
            "output": tf.reshape(
                summed_output,
                [tf.shape(summed_output)[0], 1],
                name=self.extend_name("objective_function"),
            ),
        }

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

    pgt = ParametricGeneratorTest(3, 10, 1.0)
    pgt.set_embedder_architecture([l], r, [l], r)
    pgt.set_generator_architecture([l, l], r, "tanh")

    pgt.build_input_nodes()
    pgt.build_sample_nodes()

    weights, biases = pgt.build_embedder(pgt.constraint_sample)
    generator = pgt.build_generator(pgt.latent_sample, weights, biases)
    discriminator = pgt.build_discriminator(generator["output"], pgt.constraint_sample)

    precision, recall = pgt.proxies(generator, discriminator)
    weighted_proxies = precision + pgt.proxy_weight * recall

    recall_optimiser = optimiser(recall, "recall_optimiser")
    weighted_optimiser = optimiser(weighted_proxies, "weighted_optimiser")

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    print("\nPretraining:")
    fit(
        session,
        recall_optimiser,
        None,
        10,
        20000,
        pgt.generator_training_batch_size,
        [("Recall", recall), ("Precision", precision)],
    )

    print("\nTraining:")
    fit(
        session,
        weighted_optimiser,
        None,
        50,
        20000,
        pgt.generator_training_batch_size,
        [("Recall", recall), ("Precision", precision)],
    )
