from math import log
from matplotlib import rc
import production.plots.extraction as ex
import matplotlib.pyplot as plt


def plot_satisfaction_vs_density(constraints, thinning_factor=8):
    """
    [Dict] -> Int? -> ()
    Plot the lower quartile, median, and upper quartile of the satisfaction
    probabilities of generated solutions on the y-axis, and of the relative
    densities of true solutions on the x-axis.
    """
    rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    rc("text", usetex=True)

    for constraint in constraints[::8]:
        satisfaction_probabilities, relative_densities = ex.get_constraint_statistics(
            constraint
        )
        sp_lq, sp_md, sp_uq = ex.median_with_iqr(satisfaction_probabilities)
        rd_lq, rd_md, rd_uq = ex.median_with_iqr(relative_densities)

        def process_density(d):
            return log(d + 1, 2)

        plt.errorbar(
            [process_density(rd_md)],
            [sp_md],
            xerr=([process_density(rd_md - rd_lq)], [process_density(rd_uq - rd_md)]),
            yerr=([sp_md - sp_lq], [sp_uq - sp_md]),
            fmt=".",
            color="black",
            elinewidth=1,
            capsize=3,
        )

    plt.xlabel("$\\log_2$(relative density of true solutions + 1)")
    plt.ylabel("Satisfaction probability of generated solutions")
    plt.xlim(0)
    plt.ylim(0)
    plt.show()


def training_progression_table(experiments):
    """
    [Dict] -> ()
    Produce a table of precision proxy loss and recall proxy loss from
    initialisation, through pretraining, and after training.
    """
    rows = []
    for experiment in experiments:
        rows.append(
            {
                "Experiment": len(rows) + 1,
                "Recall weight": experiment["results"]["parameters"]["recallWeight"],
                "Spread after pretraining": experiment["results"][
                    "generatorPretraining"
                ]["after"]["loss"],
                "Precision after pretraining": experiment["results"][
                    "generatorTraining"
                ]["before"]["precisionProxy"],
                "Spread after training": experiment["results"]["generatorTraining"][
                    "after"
                ]["recallProxy"],
                "Precision after training": experiment["results"]["generatorTraining"][
                    "after"
                ]["precisionProxy"],
            }
        )
    return tex_table(
        [
            "Experiment",
            "Recall weight",
            "Spread after pretraining",
            "Precision after pretraining",
            "Spread after training",
            "Precision after training",
        ],
        rows,
    )


def example_solutions_table(constraint, n_solutions):
    """
    Dict -> Int -> String
    Produce the body of a table listing a number of example solutions to the
    given constraint.
    """
    relevant_solutions = [
        solution["solution"]
        for solution in constraint["solutions"]
        if solution["type"] == "generated"
    ][:n_solutions]
    return "".join(
        [
            "&".join([format(s) for s in solution]) + "\\\\"
            for solution in relevant_solutions
        ]
    )


def tex_table(headers, rows):
    """
    [String] -> [Dict] -> [String]
    Construct a TeX table from a list of data.
    """
    strings = ["&".join(headers) + "\\\\"]
    for row in rows:
        strings.append("&".join([format(row[header]) for header in headers]) + "\\\\")
    return "".join(strings)


def format(value):
    """
    a -> String
    Return a string representation of a fundamental data type.
    """
    if isinstance(value, str):
        return value
    elif isinstance(value, int):
        return str(value)
    elif isinstance(value, float):
        return "{0:.3f}".format(value)
    else:
        raise ValueError("undefined formatting type")
