from wise.networks.deterministic.feedforwardnetwork \
    import FeedforwardNetwork
from wise.networks.activation import Activation
from wise.util.tensors import placeholder_node
from wise.util.training import classification_metrics
from wise.training.routines import fit
from wise.training.samplers.resampled import BinomialResampler
from wise.training.samplers.dataset import DataSetSampler
from environments.circles import Circles
import tensorflow as tf


class Params:
    environment = Circles
    session = tf.Session()
    # Do not include output layer shape:
    internal_layer_shapes = [[20]]
    activation = Activation.all_except_last(
        Activation.LEAKY_RELU, Activation.SIGMOID)
    save_location = None
    batch_size = 32
    fidelity=10


def make_discriminator():
    """
    () -> (tf.Placeholder, FeedforwardNetwork)
    """
    image_shape = Params.environment.image_shape(Params.fidelity)
    image_input = placeholder_node('image_input', image_shape, 1)

    return image_input, FeedforwardNetwork(
        name='pixel_discriminator',
        session=Params.session,
        input_shape=image_shape,
        layer_shapes=Params.internal_layer_shapes + [[1]],
        activations=Params.activation,
        input_node=image_input,
        save_location=Params.save_location
    )


def make_training_nodes(discriminator):
    """
    FeedforwardNetwork -> (TargetNode, LossNode, Accuracy, Optimiser)
    Create training nodes relevant to the problem.
    """
    return classification_metrics([1], discriminator.output_node,
        'discriminator_training', variables=discriminator.get_variables())


def make_sampler(image_node, satisfaction_node):
    """
    () -> FeedDictSampler
    """
    return Params.environment.pixel_environment_sampler(
        pixels_input=image_node,
        satisfaction_input=satisfaction_node,
        sampler_transform=BinomialResampler.halves_on_last_element_head
    )


def run():
    """
    () -> ()
    Attempt to train a neural network to predict the satisfaction probability
    of a continuously defined environment based on a pixel representation of
    it.
    """
    img_in, disc = make_discriminator()
    target, loss, accuracy, optimiser = make_training_nodes(disc)
    training_set_sampler = make_sampler(img_in, target)
    test_set_sampler = make_sampler(img_in, target)

    disc.get_session().run(tf.global_variables_initializer())

    fit(disc.get_session(), optimiser, training_set_sampler,
        1000, 2000, 32, [('Loss', loss), ('Accuracy', accuracy)])

    print('Validation accuracy: {}'.format(disc.feed(
        accuracy, test_set_sampler.batch(1024))))
