def get_solution_statistics(solution):
    """
    Dict -> (Float, Float, String)
    Extract satisfaction probability and relative density, as well as the solution
    type (true or generated) as a tuple from a dictionary describing them.
    """
    return (
        solution["satisfactionProbability"],
        solution["relativeDensity"],
        solution["type"],
    )


def get_constraint_statistics(constraint):
    """
    Dict -> ([Float], [Float])
    Return two lists, the first containing the satisfaction probabilities of all
    generated solutions to a logged constraint, and the second containing the
    relative densities of all true solutions to that constraint.
    """
    generated_solutions, true_solutions = partition(
        lambda s: s["type"] == "generated", constraint["solutions"]
    )
    return (
        [s["satisfactionProbability"] for s in generated_solutions],
        [s["relativeDensity"] for s in true_solutions],
    )


def get_experiment_constraint_samples(experiment):
    """
    Dict -> [Dict]
    Extract all constraint objects from a dictionary containing the results
    of a single experiment.
    """
    return experiment["results"]["evaluation"]["constraintSamples"]


def get_all_constraint_samples(experiment_repeats):
    """
    Dict -> [Dict]
    Extract all constraint objects from a dictionary containing a number
    of repeats of the same experiment.
    """
    constraints = []
    for experiment in experiment_repeats["data"]:
        constraints += get_experiment_constraint_samples(experiment)
    return constraints


def partition(predicate, values):
    """
    (a -> Bool) -> [a] -> ([a], [a])
    Partition a list of values into two lists, the first containing those values
    that satisfy the predicate, and the second those that do not.
    """
    trues, falses = [], []
    for value in values:
        if predicate(value):
            trues.append(value)
        else:
            falses.append(value)
    return trues, falses
