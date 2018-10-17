from wise.networks.deterministic.feedforwardnetwork \
    import FeedforwardNetwork
from wise.networks.activation import Activation
from wise.util.tensors import placeholder_node
from environments.circles import Circles
import tensorflow as tf


class Params:
    environment = Circles
    session = tf.Session()
    # Do not include output layer shape:
    internal_layer_shapes = [[16], [16]]
    activation = Activation.all_except_last(
        Activation.LEAKY_RELU, Activation.SIGMOID)
    save_location = None


def make_discriminator():
    """
    () -> (tf.Placeholder, tf.Placeholder, FeedforwardNetwork)
    """
    constraint_shape = Params.environment.constraint_shape()
    solution_shape = Params.environment.solution_shape()
    joint_shape = constraint_shape[:]
    joint_shape[0] += solution_shape[0]

    constraint_input = placeholder_node('constraint_input',
        constraint_shape, 1)
    solution_input = placeholder_node('solution_input',
        solution_shape, 1)
    joint_input = tf.concat([constraint_input, solution_input], 1)
    return constraint_input, solution_input, FeedforwardNetwork(
        name='artificial_discriminator',
        session=Params.session,
        input_shape=joint_shape,
        layer_shapes=Params.internal_layer_shapes + [[1]],
        activations=Params.activation,
        input_node=joint_input,
        save_location=Params.save_location
    )


def run():
    """
    () -> ()
    Attempt to train a neural network to predict the satisfaction probability
    of a continuously defined environment.  
    """
    cons_in, soln_in, disc = make_discriminator()
    disc.initialise_variables()
    print(disc.feed(disc.output_node, {cons_in: [[0, 0, 0]], soln_in: [[0.2, 0.3, 0.4]]}))
