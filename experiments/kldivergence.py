from math import pi
import tensorflow as tf


def gaussian_pdf(mean, stddev):
    """
    tf.Node -> tf.Node -> (tf.Node -> tf.Node)
    Create a partially applied function that calculates the probability
    density of a Gaussian distribution at a point with a given mean and
    standard deviation.
    """
    tau = tf.constant(2 * pi, dtype=tf.float32)
    def apply(x):
        return (1 / tf.sqrt(tau * tf.square(stddev))) * tf.exp(
            -tf.square(x - mean) / (2 * tf.square(stddev)))
    return apply


def run():
    """
    () -> ()
    Run an experiment, trying to determine whether KL divergence
    can be accurately estimated using an average of samples.
    """
    pass