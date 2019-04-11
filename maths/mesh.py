import numpy as np
import matplotlib.pyplot as plt


class Mesh:
    def __init__(self, x_range, y_range, x_bins=100, y_bins=100):
        """
        (Float, Float) -> (Float, Float) -> Int? -> Int? -> Mesh
        Create a data class for quickly sampling from a 2D grid.
        """
        self.x_range = x_range
        self.y_range = y_range
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.x_min, self.x_max = self.x_range
        self.y_min, self.y_max = self.y_range

    @property
    def x_points(self):
        """
        () -> [Float]
        Return an array of points describing each x-coordinate in one
        row of the mesh.
        """
        return np.linspace(self.x_min, self.x_max, self.x_bins)

    @property
    def y_points(self):
        """
        () -> [Float]
        Return an array of points describing each y-coordinate in one
        column of the mesh.
        """
        return np.linspace(self.y_min, self.y_max, self.y_bins)

    @property
    def all_points(self):
        """
        () -> [(Float, Float)]
        Return a list of every point in the mesh in an arbitrary order.
        """
        points = []
        for x in self.x_points:
            for y in self.y_points:
                points.append((x, y))
        return points

    @property
    def bin_size(self):
        """
        () -> (Float, Float)
        Return the bin size of the mesh.
        """
        return (
            (self.x_max - self.x_min) / self.x_bins,
            (self.y_max - self.y_min) / self.y_bins,
        )

    @property
    def half_bin_size(self):
        """
        () -> (Float, Float)
        Return half the bin size in each dimension.
        """
        x, y = self.bin_size
        return 0.5 * x, 0.5 * y

    @property
    def x_contours(self):
        """
        () -> [[(Float, Float)]]
        Return a series of lines, each consisting of points for which the
        x-value is kept constant while the y-value increases.
        """
        return [[(x, y) for y in self.y_points] for x in self.x_points]

    @property
    def y_contours(self):
        """
        () -> [[(Float, Float)]]
        Return a series of lines, each consisting of points for which the
        y-value is kept constant while the x-value increases.
        """
        return [[(x, y) for x in self.x_points] for y in self.y_points]

    def increase_density(self, times):
        """
        Int -> Mesh
        Return a new mesh which has the given number of times more bins
        in each dimension than this mesh.
        """
        return Mesh(
            self.x_range, self.y_range, self.x_bins * times, self.y_bins * times
        )

    def bound_pyplot(self):
        """
        () -> ()
        Set the bounds of the current matplotlib plot to the bounds of
        the mesh.
        """
        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)


Mesh.unit = Mesh((0.0, 1.0), (0.0, 1.0))
Mesh.double = Mesh((-1.0, 1.0), (-1.0, 1.0))
