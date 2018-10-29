import tensorflow as tf
from wise.networks.stochastic.noise import GaussianNoiseLayer
from wise.networks.network import Network


class LearnedObjectiveFunction(Network):
    """
    A collection of functions used to train a neural network to approximate
    the objective function of a problem space.
    """

    def __init__(self,
            name,
            session,
            environment,
            network_function,
            training_sampler,
            test_sampler=None,
            input_noise_stddev=None,
            save_location=None,
            ):
        """
        Create a Learned Objective Function from the given parameters.  The
        network function should be a function which takes a name, session,
        input node shape, and an input node, and returns a MLP.
        """
        # Network parameters
        super().__init__(name, session, save_location)
        self.network_function = network_function
        self.network = None

        # Environment
        self.environment = environment

        # Data
        self.training_sampler = training_sampler
        self.test_sampler = test_sampler

        # Training
        self.loss_node = None
        self.metrics = None

        # Training settings
        self.input_noise_stddev = input_noise_stddev

        # Input nodes
        self.input_placeholder = self._make_input_placeholder()
        self.transformed_input = self._make_transformed_input()

        # Network initialisation
        self.network = self.network_function(self.name, self.get_session(),
            self._get_input_shapes()[2], self.transformed_input)
        self.output_node = self.network.output_node

    def _make_input_placeholder(self):
        """
        () -> tf.Placeholder
        Create an input node which takes a representation of the environment's
        constraint and solution and feeds it into a tensorflow graph.
        """
        constraint_shape, solution_shape, joint_shape = self._get_input_shapes()
        constraint_input = placeholder_node('constraint_input',
            constraint_shape, 1)
        solution_input = placeholder_node('solution_input',
            solution_shape, 1)
        joint_input = tf.concat([constraint_input, solution_input], 1)
        return joint_input

    def _make_transformed_input(self):
        """
        () -> tf.Node
        Assuming the input placeholder exists, perform any transformations
        necessary (adding noise, etc.) and return the result.  If no transforms
        are needed, the node is returned straight away.
        """
        transformed_input = self.input_placeholder

        # Transforms
        if self.input_noise_stddev is not None:
            _, _, input_shape = self._get_input_shapes()
            transformed_input = GaussianNoiseLayer(self.extend_name('gaussian_input_noise'),
                self.get_session(), input_shape, self.input_noise_stddev,
                input_node=transformed_input)

        return transformed_input

    def _get_input_shapes(self):
        """
        () -> ([Int])
        Return the shape of the input to the constraint, solution, and then
        to the network as a whole.
        """
        constraint_shape = self.environment.constraint_shape()
        solution_shape = self.environment.solution_shape()
        joint_shape = constraint_shape[:]
        joint_shape[0] += solution_shape[0]
        return constraint_shape, solution_shape, joint_shape

    def get_variables(self):
        """
        () -> [tf.Variable]
        """
        return self.network.get_variables()
