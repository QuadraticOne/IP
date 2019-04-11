import numpy as np


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
    def all_points(self):
        """
        () -> [(Float, Float)]
        Return a list of every point in the mesh in an arbitrary order.
        """
        points = []
        for x in np.linspace(self.x_min, self.x_max, self.x_bins):
            for y in np.linspace(self.y_min, self.y_max, self.y_bins):
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


Mesh.unit = Mesh((0.0, 1.0), (0.0, 1.0))
Mesh.double = Mesh((-1.0, 1.0), (-1.0, 1.0))
