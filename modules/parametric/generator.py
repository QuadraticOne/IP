from wise.util.tensors import placeholder_node
from modules.parametric.export import ExportedParametricGenerator
import modules.reusablenet as rnet
import tensorflow as tf


class ParametricGenerator:
    def __init__(
        self,
        name,
        solution_dimension,
        latent_dimension,
        constraint_dimension,
        embedding_dimension,
    ):
        """
        Int -> Int -> Int -> Int -> ParametricGenerator
        Create a parametric generator by specifying the dimensions
        of all relevant spaces.
        """
        self.name = name
        self.extend_name = rnet.name_extender(self.name)

        self.solution_dimension = solution_dimension
        self.latent_dimension = latent_dimension
        self.constraint_dimension = constraint_dimension
        self.embedding_dimension = embedding_dimension

        # NOTE: if strange errors about non-matching dimensions are observed,
        #       it is likely because the batch size fed into the constraint node
        #       did not match this value
        self.generator_training_batch_size = 64

        self.generator_architecture_defined = False
        self.embedder_architecture_defined = False
        self.discriminator_architecture_defined = False

        self.latent_lower_bound, self.latent_upper_bound = (-1.0, 1.0)

    def set_generator_architecture(
        self, internal_layers, internal_activation, output_activation
    ):
        """
        [(Int, String)] -> String -> String -> ()
        Provide an architecture to use when building the generator.
        """
        self.generator_internal_layers = internal_layers
        self.generator_internal_activation = internal_activation
        self.generator_output_activation = output_activation

        self.generator_architecture = rnet.feedforward_network_input_dict(
            self.extend_name("generator"),
            self.latent_dimension,
            internal_layers + [(self.embedding_dimension, internal_activation)],
        )
        self.generator_architecture["layers"].append(
            {
                "input_expansion_axis": 1,
                "activation": output_activation,
                "name": "solution_embedding_layer",
            }
        )

        self.generator_architecture_defined = True

    def set_embedder_architecture(
        self,
        weights_internal_layers,
        weights_activation,
        biases_internal_layers,
        biases_activation,
    ):
        """
        [(Int, String)] -> String -> [(Int, String)] -> String -> ()
        Provide architectures to use when building the constraint
        embedders.
        """
        self.weights_internal_layers = weights_internal_layers
        self.weights_activation = weights_activation
        self.biases_internal_layers = biases_internal_layers
        self.biases_activation = biases_activation

        self.embedder_weights_architecture = rnet.feedforward_network_input_dict(
            self.extend_name("embedder_weights"),
            self.constraint_dimension,
            weights_internal_layers
            + [
                (self.embedding_dimension * self.solution_dimension, weights_activation)
            ],
        )

        self.embedder_biases_architecture = rnet.feedforward_network_input_dict(
            self.extend_name("embedder_biases"),
            self.constraint_dimension,
            biases_internal_layers + [(self.solution_dimension, biases_activation)],
        )

        self.embedder_architecture_defined = True

    def set_discriminator_architecture(self, internal_layers):
        """
        Dict -> ()
        Provide an architecture to use when building the discriminator.
        """
        self.discriminator_internal_layers = internal_layers

        self.discriminator_architecture = rnet.feedforward_network_input_dict(
            self.extend_name("discriminator"),
            self.solution_dimension + self.constraint_dimension,
            internal_layers + [(1, "sigmoid")],
        )

        self.discriminator_architecture_defined = True

    def build_input_nodes(self):
        """
        () -> ()
        Construct input placeholder nodes for the tensorflow graph.
        """
        self.solution_input = placeholder_node(
            self.extend_name("solution_input"),
            [self.solution_dimension],
            dynamic_dimensions=1,
        )
        self.latent_input = placeholder_node(
            self.extend_name("latent_input"),
            [self.latent_dimension],
            dynamic_dimensions=1,
        )
        self.constraint_input = placeholder_node(
            self.extend_name("constraint_input"),
            [self.constraint_dimension],
            dynamic_dimensions=1,
        )

    def build_sample_nodes(self):
        """
        () -> ()
        Construct sample nodes for training the generator.
        """
        self.latent_sample = self.make_latent_sample_node()
        self.constraint_sample = self.make_constraint_sample_node()

    def make_latent_sample_node(self):
        """
        () -> tf.Node
        Create an instance of a node sampling uniformly from the latent space.
        """
        return tf.random.uniform(
            [self.generator_training_batch_size, self.latent_dimension],
            minval=self.latent_lower_bound,
            maxval=self.latent_upper_bound,
            name=self.extend_name("latent_sample"),
        )

    def make_constraint_sample_node(self):
        """
        () -> tf.Node
        Create an instance of a node sampling from the constraint space in
        a manner likely to create a realistic distribution.
        """
        return tf.random.uniform(
            [self.generator_training_batch_size, self.constraint_dimension],
            minval=-1.0,
            maxval=1.0,
            name=self.extend_name("constraint_sample"),
        )

    def build_embedder(self, constraint_input):
        """
        tf.Node -> (Dict, Dict)
        Build the constraint embedder.  The output of the penultimate layer
        of the generator is assumed to have a dimensionality consistent with
        this object's `embedding_dimension` parameter.  The return values
        are separated into two networks, for the weights and biases of the
        final layer respectively.
        """
        embedder_weights_architecture = rnet.deep_copy(
            self.embedder_weights_architecture
        )
        embedder_weights_architecture["input"] = constraint_input
        weights_embedder = rnet.feedforward_network(embedder_weights_architecture)
        weights_embedder["output"] = tf.reshape(
            weights_embedder["output"],
            [
                tf.shape(weights_embedder["output"])[0],
                self.embedding_dimension,
                self.solution_dimension,
            ],
            name=self.extend_name("reshaped_constraint_embedding"),
        )

        embedder_biases_architecture = rnet.deep_copy(self.embedder_biases_architecture)
        embedder_biases_architecture["input"] = constraint_input
        biases_embedder = rnet.feedforward_network(embedder_biases_architecture)

        return weights_embedder, biases_embedder

    def build_generator(self, latent_input, weights_embedder, biases_embedder):
        """
        tf.Node -> Dict -> Dict -> Dict
        Build the generator given an input node and the weights and biases
        produced by the constraint embedder.
        """
        architecture = rnet.deep_copy(self.generator_architecture)
        architecture["input"] = latent_input
        output_layer = architecture["layers"][-1]
        output_layer["weights"] = weights_embedder["output"]
        output_layer["biases"] = biases_embedder["output"]
        return rnet.feedforward_network(architecture)

    def build_discriminator(self, solution_input, constraint_input):
        """
        tf.Node -> tf.Node -> Dict
        Build the discriminator, given nodes containing batches of
        solutions and constraints respectively.
        """
        architecture = rnet.deep_copy(self.discriminator_architecture)
        architecture["input"] = rnet.join_inputs(solution_input, constraint_input)
        discriminator = rnet.feedforward_network(architecture)
        discriminator["output"] = tf.squeeze(discriminator["output"])
        return discriminator

    def discriminator_training_nodes(self, discriminator):
        """
        Dict -> (tf.Node, tf.Node)
        Return a target node and loss node useful for training the discriminator.
        """
        target = placeholder_node(
            self.extend_name("discriminator_target"), [], dynamic_dimensions=1
        )
        loss = tf.losses.mean_squared_error(target, discriminator["output"])
        return target, loss

    def proxies(self, generator, discriminator):
        """
        Dict -> Dict -> (tf.Node, tf.Node)
        Create and return proxies for both precision and recall.
        """
        return (
            self.precision_proxy(generator, discriminator),
            self.recall_proxy(generator, discriminator),
        )

    def recall_proxy(self, generator, discriminator):
        """
        Dict -> Dict -> tf.Node
        Return a loss node that can be used to optimise a proxy for the recall of
        the generator, measuring the likelihood that a viable solution exists in
        the generated set.
        """
        if self.latent_dimension != self.solution_dimension:
            raise ValueError(
                "latent and solution dimensions must be equal "
                + "for linearity loss to be well defined"
            )
        return tf.reduce_mean(
            tf.squared_difference(
                (2 * generator["input"]) - 1,
                generator["output"],
                name=self.extend_name("linearity_error"),
            ),
            name=self.extend_name("recall_proxy"),
        )

    def precision_proxy(self, generator, discriminator):
        """
        Dict -> Dict -> tf.Node
        Return a loss node that can be used to optimise a proxy for the precision
        of the generator, measuring the likelihood of a generated sample belonging
        to the target set.
        """
        return tf.reduce_mean(
            -tf.log(
                discriminator["output"] + 1,
                name=self.extend_name("precision_logarithm"),
            ),
            name=self.extend_name("precision_proxy"),
        )

    @staticmethod
    def from_json(args):
        """
        Dict -> ParametricGenerator
        Create a parametric generator from a JSON object.
        """
        generator = ParametricGenerator(
            args["name"],
            args["solutionDimension"],
            args["latentDimension"],
            args["constraintDimension"],
            args["embeddingDimension"],
        )

        if "generatorTrainingBatchSize" in args:
            generator.generator_training_batch_size = args["generatorTrainingBatchSize"]

        if "generatorArchitecture" in args:
            gargs = args["generatorArchitecture"]
            generator.set_generator_architecture(
                ParametricGenerator._layers_from_json(gargs["internalLayers"]),
                gargs["internalActivation"],
                gargs["outputActivation"],
            )

        if "embedderArchitecture" in args:
            eargs = args["embedderArchitecture"]
            generator.set_embedder_architecture(
                ParametricGenerator._layers_from_json(
                    eargs["weights"]["internalLayers"]
                ),
                eargs["weights"]["activation"],
                ParametricGenerator._layers_from_json(
                    eargs["biases"]["internalLayers"]
                ),
                eargs["biases"]["activation"],
            )

        if "discriminatorArchitecture" in args:
            dargs = args["discriminatorArchitecture"]
            generator.set_discriminator_architecture(
                ParametricGenerator._layers_from_json(dargs)
            )

        if "latentSpace" in args:
            generator.latent_lower_bound = args["latentSpace"]["lowerBound"]
            generator.latent_upper_bound = args["latentSpace"]["upperBound"]

        return generator

    @staticmethod
    def _layers_from_json(json):
        """
        [Dict] -> [(Int, String)]
        Extract a list of tuples representing the number of nodes and activation
        of neural network layers from a list of JSON objects, each of which
        are assumed to have "nodes" and "activation" properties.
        """
        return [(layer["nodes"], layer["activation"]) for layer in json]

    @staticmethod
    def _layers_to_json(architecture):
        """
        [(Int, String)] -> [Dict]
        Convert a list of tuples describing network layers into a JSON-like object.
        """
        return [{"nodes": n, "activation": a} for n, a in architecture]

    def to_json(self):
        """
        () -> Dict
        Return a JSON-like representation of the generator's parameters.
        """
        json = {
            "solutionDimension": self.solution_dimension,
            "latentDimension": self.latent_dimension,
            "constraintDimension": self.constraint_dimension,
            "embeddingDimension": self.embedding_dimension,
            "generatorTrainingBatchSize": self.generator_training_batch_size,
        }

        if self.generator_architecture_defined:
            json["generatorArchitecture"] = {
                "internalLayers": self._layers_to_json(self.generator_internal_layers),
                "internalActivation": self.generator_internal_activation,
                "outputActivation": self.generator_output_activation,
            }

        if self.embedder_architecture_defined:
            json["embedderArchitecture"] = {
                "weights": {
                    "internalLayers": self._layers_to_json(
                        self.weights_internal_layers
                    ),
                    "activation": self.weights_activation,
                },
                "biases": {
                    "internalLayers": self._layers_to_json(self.biases_internal_layers),
                    "activation": self.biases_activation,
                },
            }

        if self.discriminator_architecture_defined:
            json["discriminatorArchitecture"] = self._layers_to_json(
                self.discriminator_internal_layers
            )

        json["latentSpace"] = {
            "lowerBound": self.latent_lower_bound,
            "upperBound": self.latent_upper_bound,
        }

        return json

    def export(self, session):
        """
        tf.Session -> ExportedParametricGenerator
        Export the generator for easy use.
        """
        return ExportedParametricGenerator(self, session)
