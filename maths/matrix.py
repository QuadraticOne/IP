import tensorflow as tf
from numpy.random import normal


class SkewSymmetricMatrix:

    def __init__(self, size):
        """Create a skew-symmetric matrix with normally distributed values."""
        self.size = size

        self.variables = tf.Variable(normal(size=[size, size]))
        self.lower_triangle = tf.linalg.band_part(self.variables, -1, 0)
        self.upper_triangle = tf.transpose(self.lower_triangle)
        self.matrix = self.lower_triangle - self.upper_triangle


class OrthogonalMatrix:

    @staticmethod
    def new(size):
        """Create an orthogonal matrix of the given size."""
        return OrthogonalMatrix(SkewSymmetricMatrix(size))

    def __init__(self, skew_symmetric_matrix):
        """Create an orthogonal matrix from a skew-symmetric matrix."""
        self.skew_symmetric_matrix = skew_symmetric_matrix
        self.matrix = tf.linalg.expm(self.skew_symmetric_matrix.matrix)
