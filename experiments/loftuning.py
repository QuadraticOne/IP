from environments.circles import Circles
from modules.discriminator import LearnedObjectiveFunction
from random import choice
from wise.util.io import IO
from wise.training.experiments.analysis import ResultsFilter, map_on_dictionary_key
from os.path import isfile
import tensorflow as tf


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


def analyse(experiment_id, save_figures=False):
    """
    String -> Bool? -> ()
    Analyse the data by plotting the progression of each architecture as its
    modules are varied across the different builder types.
    """
    pass


def extract_builder_results(experiment_id, builder_id):
    """
    String -> Int -> (String, String, [[(Float, Float)]])
    Analyse the data relating to the input module of an experiment, returning
    data to plot validation accuracy against training accuracy for each
    variation on the architectures used.

    The series is returned as a tensor whose indices, in order, are: architecture
    ID, builder option ID, 0 for training and 1 for validation.
    """
    series = []
    for architecture_id in range(N_SAMPLE_ARCHITECTURES):
        path = 'data/experiments/loftuning/{}/results/builder-{}/architecture-{}' \
            .format(experiment_id, builder_id, architecture_id)
        results = ResultsFilter(path, False)
        key = 'data.results.network_evaluations.after_training.{}.accuracy'
        series.append(results.extract_results([
            map_on_dictionary_key(key.format('training'), lambda x: x),
            map_on_dictionary_key(key.format('validation'), lambda x: x)
        ]))

    def mean(ls):
        return sum(ls) / len(ls)

    # TODO: revert the order of the last two indices
    series = list(map(lambda option_data: list(map(lambda results:
        (mean(results[0]), mean(results[1])), option_data)), series))

    return 'Training Accuracy', 'Validation Accuracy', series
