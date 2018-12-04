from environments.circles import Circles
from modules.discriminator import LearnedObjectiveFunction
from random import choice
from wise.util.io import IO
from wise.training.experiments.analysis import ResultsFilter, map_on_dictionary_key
from os.path import isfile
import tensorflow as tf
import matplotlib.pyplot as plt


# Number of different random architectures to test
N_SAMPLE_ARCHITECTURES = 10

# Number of times to repeat each experiment
REPEATS = 3

# Name of the folder insider `loftuning/` in which the run will be saved
EXPERIMENT_ID = 'initial'


training_parameters = [
    LearnedObjectiveFunction.TrainingParameters(50, 2000, 32)
]

transformed_input_builders = [
    LearnedObjectiveFunction.TransformedInputBuilder(None),
    LearnedObjectiveFunction.TransformedInputBuilder(0.01),
    LearnedObjectiveFunction.TransformedInputBuilder(0.02),
    LearnedObjectiveFunction.TransformedInputBuilder(0.05),
    LearnedObjectiveFunction.TransformedInputBuilder(0.1)
]

network_builders = [
    LearnedObjectiveFunction.NetworkBuilder([8, 8]),
    LearnedObjectiveFunction.NetworkBuilder([8, 8], bayesian=True),
    LearnedObjectiveFunction.NetworkBuilder([4]),
    LearnedObjectiveFunction.NetworkBuilder([4], bayesian=True)
]

regularisation_builders = [
    LearnedObjectiveFunction.RegularisationBuilder(l2_weight=None),
    LearnedObjectiveFunction.RegularisationBuilder(l2_weight=0.05),
    LearnedObjectiveFunction.RegularisationBuilder(l2_weight=0.1),
    LearnedObjectiveFunction.RegularisationBuilder(l2_weight=0.5),
    LearnedObjectiveFunction.RegularisationBuilder(l2_weight=1.0)
]

error_builders = [
    LearnedObjectiveFunction.ErrorBuilder(True)
]

loss_builders = [
    LearnedObjectiveFunction.LossBuilder()
]

data_builders = [
    LearnedObjectiveFunction.DataBuilder(training_set_size=64,
        validation_set_size=2048)
]

optimiser_builders = [
    LearnedObjectiveFunction.OptimiserBuilder()
]

builders = [
    training_parameters,
    transformed_input_builders,
    network_builders,
    regularisation_builders,
    error_builders,
    loss_builders,
    data_builders,
    optimiser_builders
]


def get_sample_architectures(n):
    """
    Int -> [[Builder]]
    Return a list of builder options randomly sampled from the available
    options.
    """
    return [random_parameters(builders) for _ in range(n)]


def random_parameters(options):
    """
    [[Builder]] -> [Builder]
    Take a random set of builders from a list of possible builders.
    """
    return [choice(slot_options) for slot_options in options]


def build_objective_function(pars):
    """
    [Builder] -> LearnedObjectiveFunction
    Given a list of builders, create a LearnedObjectiveFunction instance
    from those builders.
    """
    tf.reset_default_graph()
    objective_function = LearnedObjectiveFunction(
        'circles_objective_function', tf.Session(), Circles,
        pars[0], pars[1], pars[2], pars[3], pars[4], pars[5],
        pars[6], pars[7]
    )
    objective_function.feed(tf.global_variables_initializer())
    return objective_function


def vary(builder_index, architecture, repeats, subfolder):
    """
    Int -> [Builder] -> Int -> String -> ()
    For the given architecture, cycle through the available builders for the
    slot referenced by the given index.  Run the experiment `repeats` times
    for each builder.  Save the experiments in the given subfolder of
    data/experiments/loftuning/results/.  The subfolder should not start or
    end with a file path separator.
    """
    options = builders[builder_index]
    if len(options) <= 1:  # No need to run experiments if there is only one option
        return None
    option_index = 0
    for option in options:
        varied_architecture = list_with(architecture, builder_index, option)
        lof = build_objective_function(varied_architecture)
        lof.log_experiments('loftuning/{}/results/{}/option-{}'.format(
            EXPERIMENT_ID, subfolder, option_index), repeats,
            reset=lambda: lof.feed(tf.global_variables_initializer()))
        option_index += 1


def list_with(ls, index, value):
    """
    [a] -> Int -> a -> [a]
    Copy the list, replacing the element at the given index with a new value.
    """
    cp = ls[:]
    cp[index] = value
    return cp


def experiments_to_run():
    """
    () -> Int
    Determine how many experiments will be run using the current settings.
    """
    total = 0
    for options in builders:
        if len(options) > 1:
            total += len(options)
    return total * REPEATS * N_SAMPLE_ARCHITECTURES


def run():
    """
    () -> ()
    For each type of builder available to a LearnedObjectiveFunction, run
    experiments on a number of LOFs while varying only that builder.
    """
    n_experiments = experiments_to_run()
    check = input('This experiment will run {} configurations, saved in loftuning/{}/.  '
        .format(n_experiments, EXPERIMENT_ID) + 'Are you sure? (y/N) ')
    if check != 'y':
        return None

    architectures = get_sample_architectures(N_SAMPLE_ARCHITECTURES)
    architecture_index = 0
    for architecture in architectures:
        print('--- RUNNING ARCHITECTURE {}/{}'.format(
            architecture_index + 1, N_SAMPLE_ARCHITECTURES))

        lof = build_objective_function(architecture)
        io = IO('data/experiments/loftuning/{}/architectures/'.format(EXPERIMENT_ID),
            create_if_missing=True)
        io.save_json(lof.data_dictionary(), 'architecture-{}'.format(architecture_index))

        for builder_index in range(len(builders)):
            vary(builder_index, architecture, REPEATS, 'builder-{}/architecture-{}'
                .format(builder_index, architecture_index))
        architecture_index += 1


def plot_builder_validation_vs_training(experiment_id, builder_id, joined=True,
        restrict_axes=False, save_location=None):
    """
    String -> Int -> Bool? -> Bool? -> String? -> ()
    Plot validation accuracy against training accuracy for the given builder.
    If `joined` is set to False then this will produce a series of plots, as
    opposed to one plot with a line for each architecture.
    """
    key = 'data.results.network_evaluations.after_training.{}.accuracy'
    training = map_on_dictionary_key(key.format('training'), lambda x: x)
    validation = map_on_dictionary_key(key.format('validation'), lambda x: x)

    if joined:
        plot_builder_results_joined(experiment_id, builder_id, training,
            validation, 'Training accuracy', 'Validation accuracy',
            restrict_axes=restrict_axes, save_location=save_location)
    else:
        plot_builder_results_separate(experiment_id, builder_id, training,
            validation, 'Training accuracy', 'Validation accuracy', save_location)


def plot_input_builder_results(experiment_id, restrict_axes=False, save_location=None):
    """
    String -> Bool? -> String? -> ()
    Plot validation accuracy against noise standard deviation for the input
    builder options tested.
    """
    builder_id = 1
    accuracy_key = 'data.results.network_evaluations.after_training.validation.accuracy'
    stddev_key = 'data.results.parameters.input_transform.input_noise_stddev'

    plot_builder_results_joined(experiment_id, builder_id, 
        map_on_dictionary_key(stddev_key, lambda x: x or 0.0),
        map_on_dictionary_key(accuracy_key, lambda x: x),
        'Input noise standard deviation', 'Validation accuracy',
        restrict_axes=restrict_axes, save_location=save_location)


def plot_regularisation_builder_results(experiment_id,
        restrict_axes=False, save_location=None):
    """
    String -> Bool? -> String? -> ()
    Plot validation accuracy against regularisation weight for the regularisation
    builder options tested.
    """
    builder_id = 3
    accuracy_key = 'data.results.network_evaluations.after_training.validation.accuracy'
    l2_key = 'data.results.parameters.regularisation.l2_weight'

    plot_builder_results_joined(experiment_id, builder_id, 
        map_on_dictionary_key(l2_key, lambda x: x or 0.0),
        map_on_dictionary_key(accuracy_key, lambda x: x),
        'L2 weight', 'Validation accuracy',
        restrict_axes=restrict_axes, save_location=save_location)


def plot_network_builder_results(experiment_id, restrict_axes=False, save_location=None):
    """
    String -> Bool? -> String? -> ()
    Plot validation accuracy against the number of hidden nodes for the regularisation
    builder options tested.
    """
    builder_id = 2
    accuracy_key = 'data.results.network_evaluations.after_training.validation.accuracy'
    shape_key = 'data.results.parameters.network.hidden_layer_shapes'

    plot_builder_results_joined(experiment_id, builder_id, 
        map_on_dictionary_key(shape_key, lambda x: sum(x), expect_list_at_leaf=True),
        map_on_dictionary_key(accuracy_key, lambda x: x),
        'Hidden nodes', 'Validation accuracy',
        restrict_axes=restrict_axes, save_location=save_location)


def plot_builder_results_separate(experiment_id, builder_id, x_extractor, y_extractor,
        x_label, y_label, save_location=None):
    """
    String -> Int -> String -> (Dict -> a) -> (Dict -> b) -> String -> String? -> ()
    Extract two variables from the result dictionary for each option and
    architecture tested for a specific builder, then plot the value of each
    variable for each architecture on separate plots for each option tried.
    
    If a save location is provided, the plot will be saved at the given
    location instead of being displayed.  The saved location should contain
    a placeholder, '{}', that will be replaced with the option index.
    """
    results = extract_builder_results(experiment_id,
        builder_id, x_extractor, y_extractor)
    for option_index in range(len(results[0][0])):
        architecture_index = 0
        for architecture in results:
            plt.plot(architecture[0][option_index],
                architecture[1][option_index], 'o', label=str(architecture_index))
            architecture_index += 1
        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim(0.5, 1)
        plt.ylim(0.5, 1)
        if save_location is None:
            plt.show()
        else:
            plt.savefig(save_location.format(option_index))


def plot_builder_results_joined(experiment_id, builder_id, x_extractor, y_extractor,
        x_label, y_label, restrict_axes=False, save_location=None):
    """
    String -> Int -> (Dict -> a) -> (Dict -> b) -> String
        -> String -> Bool? -> String? -> ()
    Extract the values of two variables from the result dictionary of each
    option and architecture tested for a specific builder, then plot each
    architecture as a different series on a plot of the x-variable against
    the y-variable.
    
    If `restrict_axes` is set to True, both axes will be set to cover the
    range (0.5, 1).  If a save location is provided, the plot will be saved
    at the given location instead of being displayed.
    """
    results = extract_builder_results(experiment_id,
        builder_id, x_extractor, y_extractor)
    architecture_index = 0
    for architecture in results:
        plt.plot(architecture[0], architecture[1], label=str(architecture_index))
        architecture_index += 1
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if restrict_axes:
        plt.xlim(0.5, 1)
        plt.ylim(0.5, 1)
    if save_location is None:
        plt.show()
    else:
        plt.savefig(save_location)


def extract_builder_results(experiment_id, builder_id, x_extractor, y_extractor):
    """
    String -> Int -> (Dict -> a) -> (Dict -> b) -> [[[Float]]]
    Analyse the data relating to the input module of an experiment, returning
    data to plot the value indexed by the y key against that indexed by the x
    key for each variation on the architectures used.

    The series is returned as a tensor whose indices, in order, are: architecture
    ID, 0 for training and 1 for validation, builder option ID.
    """
    series = []
    for architecture_id in range(N_SAMPLE_ARCHITECTURES):
        path = 'data/experiments/loftuning/{}/results/builder-{}/architecture-{}' \
            .format(experiment_id, builder_id, architecture_id)
        results = ResultsFilter(path, False)
        series.append(results.extract_results([x_extractor, y_extractor]))

    def mean(ls):
        try:
            return sum(ls) / len(ls)
        except:
            return ls[0]

    def transpose(m):
        return [list(i) for i in zip(*m)]

    # Replace series of repeats with the mean of each series
    series = list(map(lambda option_data: list(map(lambda results:
        [mean(results[0]), mean(results[1])], option_data)), series))

    # Transpose each sub-tensor, reverting the last two indices so the list is
    # indexed (architecture, training/validation, option) instead of
    # (architecture, option, training/validation) for easier plotting
    series = list(map(transpose, series))

    return series
