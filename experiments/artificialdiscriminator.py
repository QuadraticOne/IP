from wise.networks.deterministic.feedforwardnetwork \
    import FeedforwardNetwork
from wise.networks.activation import Activation
from wise.util.tensors import placeholder_node
from wise.util.training import classification_metrics
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
    batch_size = 32


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


def make_training_nodes(discriminator):
    """
    FeedforwardNetwork -> (TargetNode, LossNode, Accuracy, Optimiser)
    Create training nodes relevant to the problem.
    """
    return classification_metrics([1], discriminator.output_node,
        'discriminator_training', variables=discriminator.get_variables())


def make_sampler(constraint_node, solution_node, satisfaction_node):
    """
    () -> FeedDictSampler
    """
    return Params.environment.environment_sampler(
        constraint_input=constraint_node,
        solution_input=solution_node,
        satisfaction_input=satisfaction_node
    )


def run():
    """
    () -> ()
    Attempt to train a neural network to predict the satisfaction probability
    of a continuously defined environment.  
    """
    cons_in, soln_in, disc = make_discriminator()
    target, loss, accuracy, optimiser = make_training_nodes(disc)
    sampler = make_sampler(cons_in, soln_in, target)

    dummy_data = [[0, 0, 0]] * Params.batch_size
    feed_dict = {cons_in: dummy_data, soln_in: dummy_data,
        target: [[0]] * Params.batch_size}
    disc.get_session().run(tf.global_variables_initializer())
    print(disc.feed([loss, accuracy], sampler.batch(128)))
    for _ in range(10000):
        disc.feed(optimiser, sampler.batch(32))
    print(disc.feed([loss, accuracy], sampler.batch(128)))
