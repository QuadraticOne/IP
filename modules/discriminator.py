import tensorflow as tf
from wise.networks.deterministic.feedforwardnetwork import FeedforwardNetwork
from wise.networks.stochastic.gaussianweightsnetwork import GaussianWeightsNetwork
from wise.networks.stochastic.noise import GaussianNoiseLayer
from wise.networks.network import Network
from wise.networks.activation import Activation
from wise.training.regularisation import l2_regularisation
from wise.training.samplers.dataset import DataSetSampler
from wise.training.samplers.resampled import BinomialResampler
from wise.training.routines import fit
from wise.training.experiments.experiment import Experiment
from wise.util.tensors import placeholder_node
from wise.util.training import classification_metrics, regression_metrics, default_adam_optimiser
from wise.util.io import IO


class LearnedObjectiveFunction(Network, Experiment):
    """
    A collection of functions used to train a neural network to approximate
    the objective function of a problem space.
    """

    def __init__(self, name, session, environment, training_parameters,
            transformed_input_builder, network_builder, regularisation_builder,
            error_builder, loss_builder, data_builder,
            optimiser_builder, save_location=None):
        """
        Create a Learned Objective Function from several parameters
        describing how it is constructed.
        """
        Network.__init__(self, name, session, save_location)
        Experiment.__init__(self, 'data/experiments/', create_folder_if_missing=True)

        self.environment = environment
        self.input_node = None

        # Parameters
        self.training_parameters = training_parameters

        # Builders
        self.input_builder = LearnedObjectiveFunction.InputBuilder()
        self.transformed_input_builder = transformed_input_builder
        self.network_builder = network_builder
        self.regularisation_builder = regularisation_builder
        self.error_builder = error_builder
        self.loss_builder = loss_builder
        self.data_builder = data_builder
        self.optimiser_builder = optimiser_builder

        self._initialise()

    def _initialise(self):
        """
        () -> ()
        """
        self.input_builder.build(self.name, self.get_session(), self.environment)
        self.transformed_input_builder.build(self.name, self.get_session(),
            self.input_builder)
        self.network_builder.build(self.name, self.get_session(),
            self.transformed_input_builder)
        self.regularisation_builder.build(self.name, self.get_session(),
            self.network_builder)
        self.error_builder.build(self.name, self.get_session(), self.network_builder)
        self.loss_builder.build(self.name, self.get_session(),
            self.regularisation_builder, self.error_builder)
        self.data_builder.build(self.name, self.get_session(), self.input_builder,
            self.error_builder, self.environment)
        self.optimiser_builder.build(self.name, self.get_session(), self.network_builder,
            self.loss_builder)

    def data_dictionary(self):
        """
        () -> Dict
        Return the properties of this learned objective function as a dictionary
        which can be logged as a JSON file.
        """
        return {
            'input': self.input_builder.data_dictionary(),
            'input_transform': self.transformed_input_builder.data_dictionary(),
            'network': self.network_builder.data_dictionary(),
            'regularisation': self.regularisation_builder.data_dictionary(),
            'error': self.error_builder.data_dictionary(),
            'loss': self.loss_builder.data_dictionary(),
            'data': self.data_builder.data_dictionary(),
            'optimiser': self.optimiser_builder.data_dictionary()
        }

    def get_variables(self):
        """
        () -> [tf.Variable]
        """
        return self.network_builder.network.get_variables()

    def evaluate_on_sampler(self, sampler, batch_size):
        """
        Sampler -> Int -> [(String, Float)]
        Evaluate each of the metrics exposed by the loss builder on a batch
        from the given sampler, then return them paired with their name.
        """
        batch = sampler.batch(batch_size)

        metrics = self.loss_builder.metrics()[:]
        outputs = self.feed([node for (name, node) in metrics], batch)
        metrics.append(('sample_size', None))
        outputs.append(batch_size)
        data = {name: output for ((name, node), output) in zip(metrics, outputs)}
        for k, v in data.items():
            data[k] = float(v)
        return data

    # Training & Experiments

    def fit(self):
        """
        () -> ()
        Run a number of epochs of training according to the parameters
        in each of the builders.
        """
        fit(self.get_session(), self.optimiser_builder.optimiser_node,
            self.data_builder.training_set_sampler, self.training_parameters.epochs,
            self.training_parameters.steps_per_epoch, self.training_parameters.batch_size,
            metrics=self.loss_builder.metrics())

    def run_experiment(self):
        """
        () -> Dict
        Run an experiment and return a dictionary containing the results for
        conversion to JSON.
        """                
        def evaluate_on_training_set():
            return self.evaluate_on_sampler(self.data_builder.training_set_sampler,
                self.training_parameters.evaluation_sample_size)

        def evaluate_on_validation_set():
            return self.evaluate_on_sampler(self.data_builder.validation_set_sampler,
                self.training_parameters.evaluation_sample_size)

        parameters = self.data_dictionary()
        before_training_training = evaluate_on_training_set()
        before_training_validation = evaluate_on_validation_set()
        self.fit()
        after_training_training = evaluate_on_training_set()
        after_training_validation = evaluate_on_validation_set()
        return {
            'parameters': parameters,
            'network_evaluations': {
                'before_training': {
                    'training': before_training_training,
                    'validation': before_training_validation
                },
                'after_training': {
                    'training': after_training_training,
                    'validation': after_training_validation
                }
            }
        }

    # Builders

    class InputBuilder:
        def __init__(self):
            self.constraint_shape = None
            self.solution_shape = None
            self.joint_shape = None
            self.environment_type = None

            self.constraint_input = None
            self.solution_input = None
            self.joint_input = None

        def build(self, name, session, environment):
            self.constraint_shape = environment.constraint_shape()
            self.solution_shape = environment.solution_shape()
            self.joint_shape = self.constraint_shape[:]
            self.joint_shape[0] += self.solution_shape[0]

            self.environment_type = environment.__name__

            self.constraint_input = placeholder_node(name + '.constraint_input',
                self.constraint_shape, 1)
            self.solution_input = placeholder_node(name + '.solution_input',
                self.solution_shape, 1)
            self.joint_input = tf.concat([self.constraint_input,
                self.solution_input], 1)

        def data_dictionary(self):
            return {
                'constraint_shape': self.constraint_shape,
                'solution_shape': self.solution_shape,
                'joint_shape': self.joint_shape,
                'environment_type': self.environment_type
            }

    class TransformedInputBuilder:
        def __init__(self, input_noise_stddev=None):
            self.input_noise_stddev = input_noise_stddev

            self.raw_input = None
            self.transformed_input = None
            self.transformed_input_shape = None

        def build(self, name, session, input_builder):
            self.raw_input = input_builder.joint_input
            self.transformed_input_shape = input_builder.joint_shape
            self.transformed_input = self.raw_input
            if self.input_noise_stddev is not None:
                self.transformed_input = GaussianNoiseLayer(name + '.gaussian_noise',
                    session, input_builder.joint_shape, self.input_noise_stddev,
                    self.transformed_input).output_node
        
        def data_dictionary(self):
            return {
                'input_noise_stddev': self.input_noise_stddev,
                'transformed_input_shape': self.transformed_input_shape
            }

    class NetworkBuilder:
        def __init__(self, hidden_layer_shapes, batch_normalisation=False,
                bayesian=False, activations=Activation.all(Activation.DEFAULT)):
            self.hidden_layer_shapes = hidden_layer_shapes
            self.batch_normalisation = batch_normalisation
            self.bayesian = bayesian
            self.activations = activations

            self.network = None
            self.output_node = None
            self.output_shape = None

        def build(self, name, session, transformed_input_builder):
            network_constructor = GaussianWeightsNetwork \
                if self.bayesian else FeedforwardNetwork
            self.network = network_constructor(name + '.network', session,
                transformed_input_builder.transformed_input_shape,
                [[s] for s in self.hidden_layer_shapes] + [[1]],
                activations=self.activations,
                input_node=transformed_input_builder.transformed_input,
                batch_normalisation=self.batch_normalisation)
            self.output_node = self.network.output_node
            self.output_shape = self.network.output_shape

        def data_dictionary(self):
            return {
                'hidden_layer_shapes': self.hidden_layer_shapes,
                'batch_normalisation': self.batch_normalisation,
                'bayesian_network': self.bayesian,
                'output_shape': self.output_shape
                # TODO: represent activations in JSON
            }

    class RegularisationBuilder:
        def __init__(self, l2_weight=None):
            self.l2_weight = l2_weight
            self.total_weight = l2_weight or 1.0

            self.l2_node = None

            self.nodes = None
            self.output_node = None

        def build(self, name, session, network_builder):
            self.nodes = []

            if self.l2_weight is not None:
                self.l2_node = self.l2_weight * l2_regularisation(network_builder.network)
                self.nodes.append(self.l2_node)

            self.output_node = tf.reduce_sum(self.nodes,
                name=name + '.regularisation_loss') / self.total_weight

        def data_dictionary(self):
            return {
                'total_weight': self.total_weight,
                'l2_weight': self.l2_weight
            }

        def metrics(self):
            all_metrics = [
                ('l2', self.l2_node)
            ]
            active_metrics = [(name, node) for (name, node) in all_metrics \
                if node is not None]
            return active_metrics

    class ErrorBuilder:
        def __init__(self, classification=True):
            self.classification = classification
            self.target_node = None
            self.error_node = None
            self.accuracy_node = None

        def build(self, name, session, network_builder):
            if self.classification:
                self.target_node, self.error_node, self.accuracy_node, _ = \
                    classification_metrics(network_builder.output_shape,
                    network_builder.output_node, name + '.classification_metrics',
                    variables=network_builder.network.get_variables())
            else:
                self.target_node, self.error_node, _ = regression_metrics(
                    network_builder.output_shape, network_builder.output_node,
                    name + '.regression_metrics', variables=
                    network_builder.network.get_variables())

        def data_dictionary(self):
            return {
                'classification': self.classification
            }

        def metrics(self):
            all_metrics = [
                ('error', self.error_node),
                ('accuracy', self.accuracy_node)
            ]
            active_metrics = [(name, node) for (name, node) in all_metrics \
                if node is not None]
            return active_metrics

    class LossBuilder:
        def __init__(self, regularisation_weight=1.0, error_weight=1.0):
            self.regularisation_weight = regularisation_weight
            self.error_weight = error_weight
            self.total_weight = self.regularisation_weight + self.error_weight

            self.loss_node = None
            self._metrics = None

        def build(self, name, session, regularisation_builder, error_builder):
            self.loss_node = tf.divide(tf.reduce_sum([
                self.regularisation_weight * regularisation_builder.output_node,
                self.error_weight * error_builder.error_node
            ]), self.total_weight, name=name + '.loss_node')
            self._metrics = regularisation_builder.metrics() + error_builder.metrics()
            self._metrics += [('total_loss', self.loss_node)]

        def data_dictionary(self):
            return {
                'total_weight': self.total_weight,
                'regularisation_weight': self.regularisation_weight,
                'error_weight': self.error_weight
            }

        def metrics(self):
            """
            () -> [(String, tf.Tensor)]
            """
            return self._metrics

    class DataBuilder:
        def __init__(self, data_folder=None, training_load_location=None,
                training_set_size=-1, validation_load_location=None,
                validation_set_size=-1):
            """
            String? -> Int? -> String? -> Int? -> DataBuilder
            To load either set as a sampler from a pickle file, provide the path
            to that file in the load location parameter, relative to a base data
            folder.  If no load location is given for a set, a set will be
            created.  A set size of -1 indicates that examples will be generated
            procedurally.
            """
            self.data_folder = data_folder
            self.training_load_location = training_load_location
            self.training_set_size = training_set_size
            self.validation_load_location = validation_load_location
            self.validation_set_size = validation_set_size

            self.root_io = IO(self.data_folder) if self.data_folder is not None else None

            self.training_set_sampler = None
            self.validation_set_sampler = None

            self.loaded_training_set = None
            self.loaded_validation_set = None
            self.procedural_training_set = None
            self.procedural_validation_set = None

        def build(self, name, session, input_builder, error_builder, environment):
            self.loaded_training_set = self.training_load_location is not None
            self.loaded_validation_set = self.validation_load_location is not None

            self.training_set_sampler = self._make_sampler(self.training_load_location,
                self.training_set_size, input_builder, error_builder, environment)
            self.validation_set_sampler = self._make_sampler(self.validation_load_location,
                self.validation_set_size, input_builder, error_builder, environment)

            self.procedural_training_set = not isinstance(
                self.training_set_sampler.sampler, DataSetSampler)
            self.procedural_validation_set = not isinstance(
                self.validation_set_sampler.sampler, DataSetSampler)

        def data_dictionary(self):
            return {
                'load_location_base': self.data_folder,
                'training_load_path': self.training_load_location,
                'validation_load_path': self.validation_load_location,
                'training_set_size': self.training_set_size,
                'validation_set_size': self.validation_set_size,
                'training_set_was_loaded': self.loaded_training_set,
                'validation_set_was_loaded': self.loaded_validation_set,
                'training_set_was_procedural': self.procedural_training_set,
                'validation_set_was_procedural': self.procedural_validation_set
            }

        def _make_sampler(self, load_location, set_size,
                input_builder, error_builder, environment):
            """
            String? -> Int -> InputBuilder -> ErrorBuilder -> Environment
                -> FeedDictSampler ([Float], [Float], [Float])
            Create a sampler from the parameters given.  If no load location is
            provided, a sampler will be made from newly generated data.  A set
            size of -1 will produce a procedural sampler, while any other set
            size will cause examples to be cached and recycled.
            """
            if load_location is not None:
                return environment.environment_sampler(
                    constraint_input=input_builder.constraint_input,
                    solution_input=input_builder.solution_input,
                    satisfaction_input=error_builder.target_node,
                    sampler_transform=lambda _: DataSetSampler.restore(
                        self.data_folder, load_location)
                )
            else:
                if set_size == -1:
                    return environment.environment_sampler(input_builder.constraint_input,
                        input_builder.solution_input, error_builder.target_node)
                else:
                    return environment.environment_sampler(
                        constraint_input=input_builder.constraint_input,
                        solution_input=input_builder.solution_input,
                        satisfaction_input=error_builder.target_node,
                        sampler_transform=lambda s: DataSetSampler.from_sampler(
                            BinomialResampler.halves_on_last_element_head(s), set_size)
                    )

        def save_samplers(self, directory, training_save_file=None,
                validation_save_file=None):
            """
            Save each of the samplers to a location as pickle files.  Each sampler
            will only be saved if a file path, relative to the base directory, is
            provided for it, and if it is not a procedural sampler.
            """
            if training_save_file is not None and not self.procedural_training_set:
                self.training_set_sampler.sampler.save(
                    directory, training_save_file)
            if validation_save_file is not None and not self.procedural_validation_set:
                self.validation_set_sampler.sampler.save(
                    directory, validation_save_file)

    class OptimiserBuilder:
        def __init__(self):
            self.optimiser_node = None

        def build(self, name, session, network_builder, loss_builder):
            self.optimiser_node = default_adam_optimiser(loss_builder.loss_node,
                name + '.optimiser', network_builder.network.get_variables())

        def data_dictionary(self):
            return {
                'optimiser': 'adam',
                'learning_rate': 0.001
            }

    class TrainingParameters:
        def __init__(self, epochs, steps_per_epoch, batch_size,
                evaluation_sample_size=2048):
            self.epochs = epochs
            self.steps_per_epoch = steps_per_epoch
            self.batch_size = batch_size

            self.evaluation_sample_size = evaluation_sample_size

            self.optimiser_node = None
