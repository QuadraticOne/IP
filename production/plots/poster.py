from production.datagen.branin import branin_function
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc, cm
import matplotlib.pyplot as plt
import numpy as np


def plot_branin_function(n=200, lower=0.0, upper=1.0):
    """
    Int? -> Float? -> Float? -> ()
    Plot the Branin function.
    """
    rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    rc("text", usetex=True)

    figure = plt.figure()
    axes = figure.gca(projection="3d")

    xs = np.linspace(lower, upper, n)
    ys = np.linspace(lower, upper, n)
    xs, ys = np.meshgrid(xs, ys)
    vectorised = np.vectorize(branin_function)
    zs = vectorised(xs, ys)
    print(zs.shape)

    surface = axes.plot_surface(
        xs, ys, zs, cmap=cm.viridis, linewidth=0, antialiased=False
    )
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_zlim(0, 1)
    plt.show()
