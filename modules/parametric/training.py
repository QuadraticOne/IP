import tensorflow as tf


def optimiser(loss, name="unnamed_optimiser"):
    """
    tf.Node -> String? -> tf.Op
    Create a default Adam optimiser for the given loss node.
    """
    return tf.train.AdamOptimizer(name=name).minimize(loss)
