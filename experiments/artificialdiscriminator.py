from wise.networks.deterministic.feedforwardnetwork \
    import FeedforwardNetwork
from wise.networks.activation import Activation
from wise.util.tensors import placeholder_node
from wise.util.training import classification_metrics
from wise.training.routines import fit
from wise.training.samplers.resampled import BinomialResampler
from wise.training.samplers.dataset import DataSetSampler
from wise.visualisation.spaceplot import evaluate_surface, plot_surface
from environments.circles import Circles
import tensorflow as tf


class Params:
    environment = Circles
    session = tf.Session()
    # Do not include output layer shape:
    internal_layer_shapes = [[4]]
    activation = Activation.all_except_last(
        Activation.LEAKY_RELU, Activation.SIGMOID)
    save_location = None
    batch_size = 32
    data_set_size = 64


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
        satisfaction_input=satisfaction_node,
        sampler_transform=lambda s: DataSetSampler.from_sampler(
            BinomialResampler.halves_on_last_element_head(s), Params.data_set_size)
    )


def run():
    """
    () -> ()
    Attempt to train a neural network to predict the satisfaction probability
    of a continuously defined environment.
    """
    cons_in, soln_in, disc = make_discriminator()
    target, loss, accuracy, optimiser = make_training_nodes(disc)
    training_set_sampler = make_sampler(cons_in, soln_in, target)
    test_set_sampler = make_sampler(cons_in, soln_in, target)

    disc.get_session().run(tf.global_variables_initializer())

    fit(disc.get_session(), optimiser, training_set_sampler,
        250, 2000, 32, [('Loss', loss), ('Accuracy', accuracy)])

    print('Validation accuracy: {}'.format(disc.feed(
        accuracy, test_set_sampler.batch(1024))))

    plot_surface(evaluate_surface(lambda x, y: Circles.solve(
        [0, 0, 0.25], [x, y, 0.25]), (-1, 1, 0.08), (-1, 1, 0.08)),
        x_label='Solution x', y_label='Solution y',
        z_label='p(satisfied | x, y)')

    plot_surface(evaluate_surface(lambda x, y: disc.feed(disc.output_node,
        {cons_in: [[0, 0, 0.25]], soln_in: [[x, y, 0.25]]})[0],
        (-1, 1, 0.08), (-1, 1, 0.08)),
        x_label='Solution x', y_label='Solution y',
        z_label='p(satisfied | x, y)')
