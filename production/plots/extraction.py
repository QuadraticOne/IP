from math import floor


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
        sorted([s["satisfactionProbability"] for s in generated_solutions]),
        sorted([s["relativeDensity"] for s in true_solutions]),
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


def percentile(p, data, presorted=False):
    """
    Float -> [Float] -> Bool -> Float
    Return the value which would sit at the pth percentile of the data.
    """
    sorted_data = sorted(data) if not presorted else data
    index = p * (len(data) - 1)
    upper_weight = index - floor(index)
    lower_weight = 1 - upper_weight
    return (lower_weight * sorted_data[floor(index)] if lower_weight > 0.0 else 0.0) + (
        upper_weight * sorted_data[floor(index + 1)] if upper_weight > 0.0 else 0.0
    )


def median_with_iqr(data):
    """
    [Float] -> (Float, Float, Float)
    Return the lower quartile, median, and upper quartile of a set of data.
    """
    sorted_data = sorted(data)
    return (
        percentile(0.25, sorted_data, presorted=True),
        percentile(0.50, sorted_data, presorted=True),
        percentile(0.75, sorted_data, presorted=True),
    )
