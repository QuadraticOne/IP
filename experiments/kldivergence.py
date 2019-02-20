from math import pi
import tensorflow as tf


def safe_log(x, eps=1e-3):
    """
    tf.Node -> Float? -> tf.Node
    Return the natural logarithm of a value with an added epsilon.
    """
    return tf.log(x + eps)


def kl_estimator(sample_p, sample_q):
    """
    tf.Node -> tf.Node -> tf.Node
    Return a node which estimates the KL divergence of p with respect
    to q, given the probability densities of a number of samples under
    p and q respectively.
    """
    return tf.reduce_mean(safe_log(sample_p / sample_q))


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
    log_term = safe_log(tf.abs(sigma_b) / tf.abs(sigma_a))
    numerator = tf.square(sigma_a) + tf.square(mu_a - mu_b)
    denominator = 2 * tf.square(sigma_b)
    return log_term + (numerator / denominator) - 0.5


def clipped_gradients(loss):
    """
    tf.Node -> tf.Operation
    Create an optimiser operation that works with clipped gradients.
    """
    optimiser = tf.train.AdamOptimizer()
    gradients = optimiser.compute_gradients(loss)
    clipped_gradients = [(tf.clip_by_value(gradient, -1., 1.), var)
        for gradient, var in gradients]
    operation = optimiser.apply_gradients(clipped_gradients)
    return operation


def run():
    """
    () -> ()
    Run an experiment, trying to determine whether KL divergence
    can be accurately estimated using an average of samples.
    """
    batch_size = 250000

    mu_a = tf.Variable(0.)
    sigma_a = tf.Variable(1.)
    mu_b = tf.constant(0.5)
    sigma_b = tf.constant(0.08)
    x = gaussian_sampler(mu_a, sigma_a, batch_size)
    p = gaussian_pdf(mu_a, sigma_a)(x)
    q = gaussian_pdf(mu_b, sigma_b)(x)

    kl = kl_estimator(p, q)
    kl_real = gaussian_divergence(mu_a, sigma_a, mu_b, sigma_b)

    optimised_node = kl if True else tf.log(kl)
    minimiser = clipped_gradients(optimised_node)

    s = tf.Session()
    s.run(tf.global_variables_initializer())

    for i in range(10000):
        output = s.run([mu_a, sigma_a, kl, kl_real, minimiser])
        if i % 150 == 0:
            print('mu: {}, sigma: {}, kl: {}, kl_real: {}'.format(
                output[0], output[1], output[2], output[3]))
