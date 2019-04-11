from matplotlib import rc
from maths.mesh import Mesh
import production.datagen.branin as branin
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
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
    set_latex_font()
    mesh = Mesh.unit

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
            mesh,
        )

    if satisfaction_cutoffs is not None:
        discriminator = export.satisfaction_probability(np.array(constraint))
        smin, smax = satisfaction_cutoffs
        draw_truth_regions(
            lambda p: smin <= discriminator(np.array([p[0], p[1]])) < smax,
            "cyan",
            0.2,
            mesh,
        )
        draw_truth_regions(
            lambda p: smax <= discriminator(np.array([p[0], p[1]])), "blue", 0.2, mesh
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

    mesh.bound_pyplot()
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


def plot_latent_contours(
    export, constraint, mesh, dimensions=(0, 1), thinning_factor=5, environment=None
):
    """
    ExportedParametricGenerator -> [Float] -> Mesh -> (Int, Int)?
        -> Int -> VectorEnvironment -> ()
    Plot lines for which l_1 and l_2 are held constant in the
    solution space.
    """
    set_latex_font()

    if environment is not None:
        draw_truth_regions(
            lambda p: environment.satisfaction(
                (np.array([p[0], p[1]]), np.array(constraint))
            ),
            "grey",
            0.2,
            mesh,
        )

    x_contours = mesh.x_contours[::thinning_factor]
    y_contours = mesh.y_contours[::thinning_factor]

    mapper = export.map_to_solution(constraint)
    x_contour_coords = [mapper(x_contour) for x_contour in x_contours]
    y_contour_coords = [mapper(y_contour) for y_contour in y_contours]

    for contour in x_contour_coords[1:]:
        xs, ys = zip(*contour)
        plt.plot(xs, ys, "blue", alpha=0.4)

    for contour in y_contour_coords[1:]:
        xs, ys = zip(*contour)
        plt.plot(xs, ys, "red", alpha=0.4)

    left, right = x_contour_coords[0], x_contour_coords[-1]

    def clip(a, b):
        return lambda x: a if x < a else (b if x > b else x)

    def plot_text(x, y, text):
        text_border_x = 0.1
        text_border_y = 0.05
        plt.text(
            clip(mesh.x_min + text_border_x, mesh.x_max - text_border_x)(x),
            clip(mesh.y_min + text_border_y, mesh.y_max - text_border_y)(y),
            text,
            ha="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1, "alpha": 0.6},
        )

    plot_text(left[0][0], left[0][1], "$l=(0,0)$")
    plot_text(left[-1][0], left[-1][1], "$l=(0,1)$")
    plot_text(right[0][0], right[0][1], "$l=(1,0)$")
    plot_text(right[-1][0], right[-1][1], "$l=(1,1)$")

    legend = []
    legend.append(
        mlines.Line2D(
            [],
            [],
            color="blue",
            marker="None",
            alpha=0.4,
            markersize=10,
            label="$l_1=\\mathrm{const}$",
        )
    )
    legend.append(
        mlines.Line2D(
            [],
            [],
            color="red",
            marker="None",
            alpha=0.4,
            markersize=10,
            label="$l_2=\\mathrm{const}$",
        )
    )
    if environment is not None:
        legend.append(
            mpatches.Patch(
                color="grey", label="$h(c, [s_1, s_2])=\\mathrm{satisfied}$", alpha=0.2
            )
        )
    plt.legend(handles=legend)

    mesh.bound_pyplot()
    plt.xlabel("$s_1$")
    plt.ylabel("$s_2$")
    plt.show()


def set_latex_font():
    """
    () -> ()
    Inform pyplot to use LaTeX default font when drawing figures.
    """
    rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    rc("text", usetex=True)
