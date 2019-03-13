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


def join_inputs(left, right):
    """Join two inputs into a single vector."""
    return tf.concat([left, right], -1)


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


def feedforward_layer_input_dict(name, input_dimension, output_dimension,
        activation, input_node=None):
    """Create an input dictionary for a reusable feedforward layer."""
    return {
        'input': tf.placeholder(tf.float32, shape=[None, input_dimension]) \
            if input_node is None else input_node,
        'weights': glorot_initialised_vars(name + '.weights',
            [input_dimension, output_dimension]),
        'biases': glorot_initialised_vars(name + '.biases', [output_dimension]),
        'activation': activation
    }


def make_layer(name, input_dimension, output_dimension,
        activation, input_node=None):
    """Create a feedforward layer from some input parameters."""
    return feedforward_layer(feedforward_layer_input_dict(name,
        input_dimension, output_dimension, activation, input_node=input_node))


def feedforward_network(input_dict):
    """
    Create a feedforward network from an input dictionary.

    The input dictionary should contain an entry for the input node,
    and a list of input dicts from which the individual layers
    can be created.  Each of these layer dictionaries will have their
    output node added, and the dictionary for the whole network will
    also have an output node added.
    """
    previous_output = input_dict['input']
    copy = {'input': input_dict['input'], 'layers': []}
    for layer in input_dict['layers']:
        layer_copy = deep_copy(layer)
        layer_copy['input'] = previous_output
        copy['layers'].append(feedforward_layer(layer_copy))
        previous_output = copy['layers'][-1]['output']
    copy['output'] = previous_output
    return copy


def feedforward_network_input_dict(name, input_dimension, layer_specs,
        input_node=None):
    """
    Create an input dictionary for a reusable feedforward network.

    Layer specifications should be given as tuples containing the number
    of hidden nodes and the activation.
    """
    _layer_specs = [layer_specs] if isinstance(layer_specs, tuple) \
        else layer_specs
    make_layer = feedforward_layer_input_dict
    return {
        'input': tf.placeholder(tf.float32, shape=[None, input_dimension]) \
            if input_node is None else input_node,
        'layers': [make_layer(name + '.' + str(i), n_in, n_out, activation) \
            for i, n_in, (n_out, activation) in zip(range(len(_layer_specs)),
                [input_dimension] + [f for f, _ in _layer_specs], _layer_specs)]
    }


def make_network(name, input_dimension, layer_specs,
        input_node=None):
    """Create a feedforward network from some input parameters."""
    return feedforward_network(feedforward_network_input_dict(name,
        input_dimension, layer_specs, input_node=input_node))
