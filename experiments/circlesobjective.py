from environments.circles import Circles
from modules.discriminator import LearnedObjectiveFunction
import tensorflow as tf


training_parameters = [
    LearnedObjectiveFunction.TrainingParameters(100, 2000, 32)
]

transformed_input_builders = [
    LearnedObjectiveFunction.TransformedInputBuilder(None)
]

network_builders = [
    LearnedObjectiveFunction.NetworkBuilder([8],
        bayesian=False, batch_normalisation=False)
]

regularisation_builders = [
    LearnedObjectiveFunction.RegularisationBuilder(l2_weight=None)
]

error_builders = [
    LearnedObjectiveFunction.ErrorBuilder(True)
]

loss_builders = [
    LearnedObjectiveFunction.LossBuilder()
]

data_builders = [
    LearnedObjectiveFunction.DataBuilder()
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


def combinations(lists):
    """
    [[a]] -> [[a]]
    Given a list of lists, with each sub list containing the possible candidates
    for that index in the main list, return a list of combinations of the ways
    in which the candidates can be combined in the main list.
    """
    indices = iterate_tensor_shapes([len(builder_list) for builder_list in lists])
    return [[builder_list[i] for i, builder_list in zip(index, lists)] \
        for index in indices]


def iterate_tensor_shapes(shape):
    """
    [Int] -> [[Int]]
    Return a list of all the indices in the tensor of the given shape.
    """
    rank = len(shape)
    indices = []
    current_index = [0] * rank
    while current_index[-1] < shape[-1]:
        indices.append(current_index[:])
        i = 0
        for i in range(rank):
            current_index[i] += 1
            if current_index[i] == shape[i] and i < rank - 1:
                current_index[i] = 0
            else:
                break
        print(indices[-1])
    return indices


def run():
    """
    () -> ()
    Train a LearnedObjectiveFunction on the Circles environment, using a number
    of different combinations of builders for each component of the experiment's
    architecture.  The different possible builders are defined in separate lists
    - one for each type of builder - and running the experiment will perform
    and log one for each possible combination under the experiments/circlesobjective/
    folder.  Be aware that the number of experiments run will be proportional to
    the product of the lengths of all lists, which may increase exponentially.
    """
    i = 0
    parameter_combinations = combinations(builders)
    for pars in parameter_combinations:
        objective_function = LearnedObjectiveFunction(
            'circles_objective_function', tf.Session(), Circles,
            pars[0], pars[1], pars[2], pars[3], pars[4], pars[5],
            pars[6], pars[7]
        )
        objective_function.feed(tf.global_variables_initializer())
        objective_function.log_experiment('circlesobjective/comination-{}'
            .format(i))
        i += 1
