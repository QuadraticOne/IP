import tensorflow as tf


class Metrics:
    def __init__(self, trainer, generator=None, embedder=None, discriminator=None):
        """
        Trainer -> Dict? -> Dict? -> Dict? -> Metric
        Provides easy access to a number of metrics and loss functions.
        """
        self.trainer = trainer

        self.generator = generator
        self.embedder = embedder
        self.discriminator = discriminator

        self.metrics = {
            "precision": self.precision,
            "uniformity": self.uniformity,
            "separation": self.separation,
        }

    def get(self, identifier, metadata={}):
        """
        String -> tf.Node
        Return a tensorflow node that evaluates the named metric, if it exists.
        """
        if identifier not in self.metrics:
            raise ValueError(
                "metric identifier should be one of {} but is '{}'".format(
                    ", ".join(["'{}'".format(m) for m in self.metrics.keys()]),
                    identifier,
                )
            )
        return self.metrics[identifier](metadata)

    def require(self, generator=False, embedder=False, discriminator=False):
        """
        Bool? -> Bool? -> Bool? -> ()
        Throw errors if any of the specified networks are undefined.
        """
        if generator and self.generator is None:
            raise ValueError("generator should be defined but is not")
        if embedder and self.embedder is None:
            raise ValueError("embedder should be defined but is not")
        if discriminator and self.discriminator is None:
            raise ValueError("discriminator should be defined but is not")

    def precision(self, _):
        """
        Dict -> tf.Node
        Return a node that computes the precision of the samples taken from the
        generator.
        """
        self.require(discriminator=True)
        return tf.reduce_mean(
            -tf.log(
                self.discriminator["output"] + self.trainer.epsilon,
                name="precision_logarithm",
            ),
            name="precision",
        )

    def uniformity(self, _):
        """
        Dict -> tf.Node
        Return a node that represents the distance of the generator from the identity
        function in the function space.
        """
        self.require(generator=True)
        g = self.trainer.parametric_generator
        if g.latent_dimension != g.solution_dimension:
            raise ValueError(
                "latent and solution dimensions must be equal "
                + "for uniformity to be well defined"
            )
        return tf.reduce_mean(
            tf.squared_difference(
                map_between(
                    (g.latent_lower_bound, g.latent_upper_bound),
                    (g.solution_lower_bound, g.solution_upper_bound),
                    self.generator["input"],
                ),
                self.generator["output"],
                name="uniformity_error",
            ),
            name="uniformity",
        )

    def separation(self, metadata):
        """
        Dict -> tf.Node
        Return a node that calculates the mean distance between two points in the
        generator sample.
        """
        self.require(generator=True)
        target = metadata["target"] if "target" in metadata else 1.0

        samples = self.generator["output"]
        shape = tf.shape(samples)
        repeated = tf.tile(samples, [shape[0], 1])
        grouped = tf.reshape(
            tf.tile(samples, [1, shape[0]]), [shape[0] * shape[0], shape[1]]
        )
        squared_difference = tf.square(repeated - grouped)
        mean_separation = tf.reduce_mean(squared_difference)
        return tf.square(target - mean_separation)

    def satisfaction_probability(self, metadata):
        """
        Dict -> tf.Node
        Return a node that calculates the probability that a sample from the latent
        space, when mapped to the solution space, will satisfy its constraints.  The
        first argument is the cutoff probability, below which the solution will not
        count as satisfying the constraints.
        """
        self.require(discriminator=True)
        cutoff = metadata["cutoff"] if "cutoff" in metadata else 0.8
        return tf.reduce_mean(
            tf.where(tf.less(self.discriminator["output"], cutoff), 0.0, 1.0)
        )


def head_and_tail(values):
    """
    [a] -> (a, [a])
    Split a list into its head and tail, assuming it has at least one
    value.  The tail may be an empty list.
    """
    return values[0], values[1:]


def map_between(input_bounds, output_bounds, node):
    """
    (Float, Float) -> (Float, Float) -> tf.Node -> tf.Node
    Map each dimension of a tensorflow node from one range to another, linearly.
    """
    input_range = input_bounds[1] - input_bounds[0]
    output_range = output_bounds[1] - output_bounds[0]
    scaled_input = (node - input_bounds[0]) / input_range
    return output_bounds[0] + (output_range * scaled_input)
