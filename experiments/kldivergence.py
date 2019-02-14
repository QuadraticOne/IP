from math import pi
import tensorflow as tf


def kl_estimator(sample_p, sample_q):
    """
    tf.Node -> tf.Node -> tf.Node
    Return a node which estimates the KL divergence of p with respect
    to q, given the probability densities of a number of samples under
    p and q respectively.
    """
    return tf.reduce_mean(tf.log(sample_p / sample_q))


def gaussian_sampler(mean, stddev, batch_size):
    """
    tf.Node -> tf.Node -> Int -> tf.Node
    Create a node of shape [batch_size] which samples from a Gaussian
    probability distribution.
    """
    return tf.random_normal([batch_size], mean=mean, stddev=stddev)


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


def gaussian_divergence(mu_a, sigma_a, mu_b, sigma_b):
    """
    tf.Node -> tf.Node -> tf.Node -> tf.Node -> tf.Node
    Return a node that calculates the KL divergence of two Gaussian
    probability distributions.
    """
    log_term = tf.log(sigma_b / sigma_a)
    numerator = tf.square(sigma_a) + tf.square(mu_a - mu_b)
    denominator = 2 * tf.square(sigma_b)
    return log_term + (numerator / denominator) - 0.5


def run():
    """
    () -> ()
    Run an experiment, trying to determine whether KL divergence
    can be accurately estimated using an average of samples.
    """
    pass
