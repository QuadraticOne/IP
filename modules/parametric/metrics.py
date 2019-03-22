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

        self.metrics = {"precision": self.precision, "uniformity": self.uniformity}

    def get(self, metric_name):
        """
        String -> tf.Node
        Return a tensorflow node that evaluates the named metric, if it exists.
        """
        if metric_name not in self.metrics:
            raise ValueError(
                "metric should be one of {} but is '{}'".format(
                    ", ".join(["'{}'".format(m) for m in self.metrics.keys()]),
                    metric_name,
                )
            )
        return self.metrics[metric_name]()

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

    def precision(self):
        """
        () -> tf.Node
        Return a node that computes the precision of the samples taken from the
        generator.
        """
        self.require(discriminator=True)
        return tf.reduce_mean(
            -tf.log(self.discriminator["output"] + 1, name="precision_logarithm"),
            name="precision",
        )

    def uniformity(self):
        """
        () -> tf.Node
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
                (2 * self.generator["input"]) - 1,
                self.generator["output"],
                name="uniformity_error",
            ),
            name="uniformity",
        )
