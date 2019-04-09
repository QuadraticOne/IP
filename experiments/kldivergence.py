from math import pi
import tensorflow as tf


class Args:

    data_type = tf.float32
    gradient_cutoff = -1
    use_linearised_log = False


def safe_log(x, eps=1e-3):
    """
    tf.Node -> Float? -> tf.Node
    Return the natural logarithm of a value with an added epsilon.
    """
    if Args.use_linearised_log:
        eps_node = tf.constant(eps, dtype=Args.data_type)
        linearised = x / eps_node + (tf.log(eps_node) - 1)
        return tf.where(tf.less(x, eps), linearised, tf.log(x))
    else:
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
    return tf.random_normal(
        [batch_size], mean=mean, stddev=stddev, dtype=Args.data_type
    )


def gaussian_pdf(mean, stddev):
    """
    tf.Node -> tf.Node -> (tf.Node -> tf.Node)
    Create a partially applied function that calculates the probability
    density of a Gaussian distribution at a point with a given mean and
    standard deviation.
    """
    tau = tf.constant(2 * pi, dtype=Args.data_type)

    def apply(x):
        return (1 / tf.sqrt(tau * tf.square(stddev))) * tf.exp(
            -tf.square(x - mean) / (2 * tf.square(stddev))
        )

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
    clipped_gradients = [
        (tf.clip_by_value(gradient, -Args.gradient_cutoff, Args.gradient_cutoff), var)
        for gradient, var in gradients
    ]
    operation = optimiser.apply_gradients(clipped_gradients)
    return operation


def run(
    initial_mean, initial_stddev, batch_size, epochs, evaluation_frequency, log=False
):
    """
    Float -> Float -> Int -> Int -> Int -> Bool? -> [Dict]
    Run an experiment, trying to determine whether KL divergence
    can be accurately estimated using an average of samples.
    """
    mu_a = tf.Variable(initial_mean, dtype=Args.data_type)
    sigma_a = tf.Variable(initial_stddev, dtype=Args.data_type)
    mu_b = tf.constant(0.0, dtype=Args.data_type)
    sigma_b = tf.constant(1.0, dtype=Args.data_type)
    x = gaussian_sampler(mu_a, sigma_a, batch_size)
    p = gaussian_pdf(mu_a, sigma_a)(x)
    q = gaussian_pdf(mu_b, sigma_b)(x)

    kl = kl_estimator(p, q)
    kl_real = gaussian_divergence(mu_a, sigma_a, mu_b, sigma_b)

    optimised_node = kl
    minimiser = (
        clipped_gradients(optimised_node)
        if Args.gradient_cutoff > 0
        else (tf.train.AdamOptimizer().minimize(optimised_node))
    )

    s = tf.Session()
    s.run(tf.global_variables_initializer())

    data = []

    for i in range(epochs):
        output = s.run([mu_a, sigma_a, kl, kl_real, minimiser])
        if i % evaluation_frequency == 0:
            if log:
                print(
                    "mu: {}, sigma: {}, kl: {}, kl_real: {}".format(
                        output[0], output[1], output[2], output[3]
                    )
                )
            data.append({"epoch": i, "mean": output[0], "stddev": output[1]})

    return data
