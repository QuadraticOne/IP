import tensorflow as tf
from wise.networks.stochastic.noise import GaussianNoiseLayer
from wise.networks.network import Network
from wise.util.tensors import placeholder_node


class LearnedObjectiveFunction(Network):
    """
    A collection of functions used to train a neural network to approximate
    the objective function of a problem space.
    """

    def __init__(self, name, session, environment, transformed_input_builder,
            network_builder, regularisation_builder, error_builder,
            loss_builder, data_builder, save_location=None):
        """
        Create a Learned Objective Function from several parameters
        describing how it is constructed.
        """
        super().__init__(name, session, save_location)

        self.environment = environment
        self.input_node = None

        # Builders
        self.input_builder = LearnedObjectiveFunction.InputBuilder()
        self.transformed_input_builder = transformed_input_builder
        self.network_builder = network_builder
        self.regularisation_builder = regularisation_builder
        self.error_builder = error_builder
        self.loss_builder = loss_builder
        self.data_builder = data_builder

        self._initialise()

    def _initialise(self):
        """
        () -> ()
        """
        self.input_builder.build(self.name, self.get_session(), self.environment)

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

            self.constraint_input = placeholder_node('constraint_input',
                self.constraint_shape, 1)
            self.solution_input = placeholder_node('solution_input',
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
        pass

    class NetworkBuilder:
        pass

    class RegularisationBuilder:
        pass

    class ErrorBuilder:
        pass

    class LossBuilder:
        pass

    class DataBuilder:
        pass
