from random import uniform
import tensorflow as tf
import numpy as np


def partition_randomly(data, p_true):
    """
    [a] -> Float -> ([a], [a])
    Divide data randomly into two groups, with the probability of belonging to 
    the first group specified.
    """
    trues, falses = [], []
    for datum in data:
        if uniform(0, 1) < p_true:
            trues.append(datum)
        else:
            falses.append(datum)
    return trues, falses


def training_input_nodes(n_points, batch_size, validation_proportion=0.2):
    """
    Int -> Int -> Float? -> ([a] -> (tf.Node, tf.Node))
    Create a generator that makes sample nodes for training and validation for a
    dataset, given the size of the dataset and the required batch size.
    """
    training_indices, validation_indices = partition_randomly(
        range(n_points), 1 - validation_proportion
    )
    batch_indices = tf.random.uniform(
        [batch_size], maxval=len(training_indices), dtype=tf.int32
    )
    return sample_and_validation_nodes(
        training_indices, validation_indices, batch_indices
    )


def sample_and_validation_nodes(training_indices, validation_indices, batch_indices):
    """
    [Int] -> [Int] -> tf.Node Int -> ([a] -> (tf.Node, tf.Node))
    Create a generator that makes sample nodes for training and validation from 
    the given dataset.
    """

    def generate_nodes(data):
        training_constant = tf.constant(np.array([data[i] for i in training_indices]))
        validation_constant = tf.constant(
            np.array([data[i] for i in validation_indices])
        )
        training_sample = tf.nn.embedding_lookup(training_constant, batch_indices)
        return training_sample, validation_constant

    return generate_nodes
