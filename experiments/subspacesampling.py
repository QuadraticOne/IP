from wise.networks.deterministic.feedforwardnetwork import FeedforwardNetwork
from wise.networks.activation import Activation
from wise.util.training import default_adam_optimiser
from wise.training.routines import fit
from numpy import linspace
from os import makedirs
from maths.mcmc import metropolis_hastings
import matplotlib.pyplot as plt
import tensorflow as tf


class Args:

    n = 1
    batch_size = 256
    session = tf.Session()
    g_hidden = [[8], [8]]
    w = 5
    output_activation = Activation.TANH
    target_spread = 1.0


def uniform_node():
    """
    () -> tf.Node
    Create a node that represents a list of samples from an n-dimensional
    unit hypercube.
    """
    return tf.random.uniform([Args.batch_size, Args.n])


def g(y):
    """
    tf.Node -> tf.Node
    Create a function that maps from the latent space S_L to the
    generated space S_G.
    """
    return FeedforwardNetwork(
        'g', Args.session, [Args.n], Args.g_hidden + [[1]],
        activations=Activation.all_except_last(
            Activation.LEAKY_RELU, Args.output_activation),
        input_node=y).output_node


def f(x):
    """
    tf.Node -> tf.Node
    Create a Tensorflow graph representing the objective function, which
    is expected to output values between 0 and 1.
    """
    return sigmoid_bump(x, width=Args.w, offset=0.5) + \
        sigmoid_bump(x, width=Args.w, offset=-0.5)


def sigmoid_bump(x, width=4, offset=0., fatness=0.05, y_scale=1.):
    """
    tf.Node -> Float -> Float? -> Float? -> Float? -> tf.Node
    Create a Tensorflow graph representing a sigmoid bump.
    """
    after_offset = (x - offset) / fatness
    return y_scale * (tf.sigmoid(after_offset + width) -
        tf.sigmoid(after_offset - width))


def p_loss(gamma):
    """
    tf.Node -> tf.Node
    Create a node that estimates the a proxy loss for maximising the
    precision of the generator network.
    """
    return tf.reduce_mean(-tf.log(gamma + 1))


def plot_histogram(node, lower=None, upper=None, steps=50, show=False, save=None):
    """
    tf.Node -> Float? -> Float? -> Int? -> Bool? -> String? -> ()
    Plot a histogram of the given node, where the node is assumed to be
    a list of vectors, and the first component of each vector is used.
    """
    values = sorted([v[0] for v in Args.session.run(node)])
    l = lower if lower is not None else values[0]
    u = upper if upper is not None else values[-1]
    plt.hist(values, bins=steps, range=(l, u))
    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.cla()


def f_plotter(lower, upper, steps=50):
    """
    () -> (() -> ())
    Return a function which, when called, plots the objective function
    for the specified range.
    """
    xs = linspace(lower, upper, steps)
    fs = Args.session.run(f(tf.constant(xs)))
    def plot(show=False, save=None):
        plt.plot(xs, fs)
        if show:
            plt.show()
        if save is not None:
            plt.savefig(save)
        plt.cla()
    return plot


def plot_latent_relation(latent_samples, solution_samples, show=False, save=None):
    """
    tf.Node -> tf.Node -> Bool? -> String? -> ()
    Plot the relationship between the latent space and solution space, assuming
    both are one-dimensional.
    """
    xs, ys = Args.session.run([solution_samples, latent_samples])
    plt.plot(xs, ys, '.')
    plt.xlabel('Solution value')
    plt.ylabel('Latent value')
    plt.xlim(-1, 1)
    plt.ylim(0, 1)
    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.cla()


def mcmc_samples(distribution_input, distribution_output,
        tensorflow_session, n_samples, skip, start):
    """
    tf.Node -> tf.Node -> tf.Session -> Int -> Int -> [Float] -> [[Float]]
    Take a number of samples from a target distribution, as defined by a
    tensorflow node.
    """
    samples = [start]
    for _ in range(n_samples):
        samples.append(metropolis_hastings(skip,
            lambda x: tensorflow_session.run(distribution_output,
                feed_dict={distribution_input: x}),
            samples[-1]))
    return samples


def sample_target_distribution(dimensions, samples, skip):
    """
    Int -> Int -> Int -> [[Float]]
    Take a number of samples from the target distribution using the
    Metropolis-Hastings algorithm.
    """
    pdf_input = tf.placeholder(tf.float32, shape=[dimensions])
    pdf_output = f(pdf_input)
    session = tf.Session()
    return mcmc_samples(pdf_input, pdf_output, session, samples, skip,
        [0.0] * dimensions)


def spread(samples):
    """
    Calculate the mean squared distance between each pair of samples.

    Takes a (b, n) tensor and splits it into two (b * b, n) tensors such
    that, when lined up, each sample is paired with a different sample
    from the same set.  Then calculates the L2 distance between each of
    the pair and returns the mean of the result.
    """
    shape = tf.shape(samples)
    repeated = tf.tile(samples, [shape[0], 1])
    grouped = tf.reshape(tf.tile(samples, [1, shape[0]]),
        [shape[0] * shape[0], shape[1]])
    squared_difference = tf.square(repeated - grouped)
    return tf.reduce_mean(squared_difference)


def mean_magnitude_squared(samples):
    """Calculate the mean square of the magnitude of a number of samples."""
    return tf.reduce_mean(tf.square(samples))


def run():
    """
    () -> ()
    Attempt to learn a function which maps samples from a unit hypercube
    to another space defined as the values of x for which f(x) > 0.5.
    """
    y_sample = uniform_node()
    x_sample = g(y_sample)
    x_spread = spread(x_sample)
    spread_error = 0.5 * tf.square(Args.target_spread - x_spread)
    gamma_sample = f(x_sample)
    precision = p_loss(gamma_sample)
    maximise_precision = default_adam_optimiser(precision, 'precision_optimiser')
    optimise_spread = default_adam_optimiser(spread_error, 'mean_minimiser')
    balance = default_adam_optimiser(1. * precision + spread_error, 'balancer')

    run_id = input('Enter run ID: ')
    loc = 'figures/subspacesampling/onedimensional/' + run_id + '/'
    makedirs(loc)

    plot_f = f_plotter(-1, 1)
    plot_f(save=loc + 'objective_function')

    def plot_x_histogram(show=False, save=None):
        plot_histogram(x_sample, lower=-1, upper=1, show=show, save=save)

    def plot_gamma_histogram(show=False, save=None):
        plot_histogram(gamma_sample, lower=0, upper=1, show=show, save=save)

    Args.session.run(tf.global_variables_initializer())

    plot_latent_relation(y_sample, x_sample, save=loc + 'y_vs_x_before')
    plot_x_histogram(save=loc + 'x_before')
    plot_gamma_histogram(save=loc + 'gamma_before')

    for i in range(1000):
        precision_loss, spread_loss, _ = Args.session.run(
            [precision, x_spread, optimise_spread])
        if i % 100 == 0:
            print('Precision loss: {}\tSpread loss: {}'.format(
                precision_loss, spread_loss))

    plot_latent_relation(y_sample, x_sample, save=loc + 'y_vs_x_intermediate')
    plot_x_histogram(save=loc + 'x_intermediate')
    plot_gamma_histogram(save=loc + 'gamma_intermediate')

    for i in range(2000):
        precision_loss, spread_loss, _ = Args.session.run(
            [precision, x_spread, balance])
        if i % 100 == 0:
            print('Precision loss: {}\tSpread loss: {}'.format(
                precision_loss, spread_loss))

    plot_latent_relation(y_sample, x_sample, save=loc + 'y_vs_x_after')
    plot_x_histogram(save=loc + 'x_after')
    plot_gamma_histogram(save=loc + 'gamma_after')
