from wise.util.tensors import glorot_initialised_vars
import tensorflow as tf


def deep_copy(value):
    """Create a deep copy of a JSON-like object."""
    if isinstance(value, dict):
        return {k: deep_copy(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [deep_copy(v) for v in value]
    else:
        return value


def feedforward_layer(input_dict):
    """
    Create a feedforward layer from an input dictionary.
    
    The input dictionary should contain entries, in the form of
    tensorflow nodes, for the input, weights, and biases.  A new
    entry will be added for the output node.  The activation should
    be specified as a string; one of 'leaky-relu', 'relu', 'sigmoid',
    or 'tanh'.
    """
    activation = {
        'sigmoid': tf.nn.sigmoid,
        'relu': tf.nn.relu,
        'leaky-relu': tf.nn.leaky_relu,
        'tanh': tf.nn.tanh
    }[input_dict['activation']]
    copy = deep_copy(input_dict)
    copy['output'] = activation(tf.matmul(
        input_dict['input'], input_dict['weights']) + input_dict['biases'])
    return copy


def feedforward_layer_input_dict(input_dimension, output_dimension,
        activation, input_node=None):
    """Create an input dictionary for a reusable feedforward layer."""
    return {
        'input': tf.placeholder(tf.float32, shape=[None, input_dimension]),
        'weights': glorot_initialised_vars('weights',
            [input_dimension, output_dimension]),
        'biases': glorot_initialised_vars('biases', [output_dimension]),
        'activation': activation
    }
