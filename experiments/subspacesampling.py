from wise.networks.deterministic.feedforwardnetwork import FeedforwardNetwork
from wise.networks.activation import Activation
from wise.util.training import default_adam_optimiser
from wise.training.routines import fit
import tensorflow as tf


class Args:

    n = 1
    batch_size = 32
    session = tf.Session()
    g_hidden = [[8]]
    w = 0.5


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
            Activation.LEAKY_RELU, Activation.TANH),
        input_node=y).output_node


def f(x):
    """
    tf.Node -> tf.Node
    Create a Tensorflow graph representing the objective function, which
    is expected to output values between 0 and 1.
    """
    return 0.5 * (tf.sigmoid(x + Args.w) - tf.sigmoid(x - Args.w))


def p_loss(gamma):
    """
    tf.Node -> tf.Node
    Create a node that estimates the a proxy loss for maximising the
    precision of the generator network.
    """
    return tf.reduce_mean(-tf.log(gamma))


def run():
    """
    () -> ()
    Attempt to learn a function which maps samples from a unit hypercube
    to another space defined as the values of x for which f(x) > 0.5.
    """
    y_sample = uniform_node()
    x_sample = g(y_sample)
    gamma_sample = f(x_sample)
    l = p_loss(gamma_sample)
    opt = default_adam_optimiser(l, 'optimiser')

    Args.session.run(tf.global_variables_initializer())

    print(Args.session.run(x_sample))

    for i in range(1000):
        epoch_loss, _ = Args.session.run([l, opt])
        if i % 10 == 0:
            print(epoch_loss)

    print(Args.session.run(x_sample))
