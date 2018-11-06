training_parameters = [

]

input_builders = [

]

transformed_input_builders = [

]

network_builders = [

]

regularisation_builders = [

]

error_builders = [

]

loss_builders = [

]

data_builders = [

]

optimiser_builders = [

]


def combinations(lists):
    """
    [[a]] -> [[a]]
    Given a list of lists, with each sub list containing the possible candidates
    for that index in the main list, return a list of combinations of the ways
    in which the candidates can be combined in the main list.
    """
    pass


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
    pass
