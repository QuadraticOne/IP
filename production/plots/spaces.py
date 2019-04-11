from matplotlib import rc
from maths.mesh import Mesh
import production.datagen.branin as branin
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


eps = 1e-4


def plot_branin_solutions(
    export, constraint, n_samples=1024, satisfaction_cutoffs=(0.7, 0.9)
):
    """
    ExportedParametricGenerator -> [Float] -> Int? -> (Float, Float)? -> ()
    Plot a set of sampled solutions for the Branin function environment.
    """
    plot_solutions(
        export,
        constraint,
        n_samples=n_samples,
        satisfaction_cutoffs=satisfaction_cutoffs,
        environment=branin.branin,
    )


def plot_solutions(
    export,
    constraint,
    n_samples,
    dimensions=(0, 1),
    environment=None,
    satisfaction_cutoffs=None,
):
    """
    ExportedParametricGenerator -> [Float] -> Int -> (Int, Int)?
        -> VectorEnvironment? -> (Float, Float)? -> ()
    Plot a number of solutions sampled for a constraint in
    two-dimensional space.  If a constraint satisfaction function is provided
    it will also be plotted for the given constraint, and another flag
    can be set to plot the learned viable set according to the discriminator
    when the satisfaction probability is predicted to be above some cutoff.
    """
    rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    rc("text", usetex=True)

    solutions = [
        s.solution
        for s in export.sample_for_constraint(np.array(constraint), n_samples)
    ]

    x, y = dimensions
    xs = [s[x] for s in solutions]
    ys = [s[y] for s in solutions]

    if environment is not None:
        draw_truth_regions(
            lambda p: environment.satisfaction(
                (np.array([p[0], p[1]]), np.array(constraint))
            ),
            "red",
            0.2,
            Mesh.unit,
        )

    if satisfaction_cutoffs is not None:
        discriminator = export.satisfaction_probability(np.array(constraint))
        smin, smax = satisfaction_cutoffs
        draw_truth_regions(
            lambda p: smin <= discriminator(np.array([p[0], p[1]])) < smax,
            "cyan",
            0.2,
            Mesh.unit,
        )
        draw_truth_regions(
            lambda p: smax <= discriminator(np.array([p[0], p[1]])),
            "blue",
            0.2,
            Mesh.unit,
        )

    plt.plot(xs, ys, ".")

    legend = []
    legend.append(
        mlines.Line2D(
            [],
            [],
            color="blue",
            marker=".",
            linestyle="None",
            markersize=10,
            label="Sample solution",
        )
    )

    if environment is not None:
        legend.append(
            mpatches.Patch(
                color="red", label="$h(c, [s_1, s_2])=\\mathrm{satisfied}$", alpha=0.2
            )
        )
    if satisfaction_cutoffs is not None:
        legend.append(
            mpatches.Patch(
                color="cyan",
                label="${}\le h'(c, [s_1, s_2])<{}$".format(smin, smax),
                alpha=0.2,
            )
        )
        legend.append(
            mpatches.Patch(
                color="blue", label="$h'(c, [s_1, s_2])\ge{}$".format(smax), alpha=0.2
            )
        )

    plt.legend(handles=legend)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("$s_{}$".format(x + 1))
    plt.ylabel("$s_{}$".format(y + 1))
    plt.show()


def draw_truth_regions(f, colour, alpha, mesh, show=False):
    """
    ((Float, Float) -> Bool) -> String -> Float -> Mesh -> Bool? -> ()
    Get the truth regions for a function on a given domain and then fill
    them in on the current plot.  Optionally, show the plot afterwards.
    """
    draw_boxes(get_truth_regions(f, mesh), colour, alpha, mesh, show=show)


def get_truth_regions(f, mesh):
    """
    ((Float, Float) -> Bool) -> Mesh -> [(Float, Float)]
    Return a list of points for which the given function returns True.
    """
    return [point for point in mesh.all_points if f(point)]


def draw_boxes(points, colour, alpha, mesh, show=False):
    """
    [(Float, Float)] -> String -> Float -> Mesh -> Bool? -> ()
    Add shaded pixels to the current plot in squares of equal size centred
    on each of the given points.  Optionally also draws the plot afterwards.
    """
    dx, dy = mesh.half_bin_size

    for x, y in points:
        plt.fill(
            [x - dx, x - dx, x + dx + eps, x + dx + eps],
            [y - dx, y + dx + eps, y + dx + eps, y - dx],
            colour,
            alpha=alpha,
        )

    if show:
        plt.show()
