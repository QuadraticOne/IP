class NTree:
    def __init__(self, ranges, split_dimension, points, capacity, parent=None):
        """
        [(Float, Float)] -> Int -> [[Float]] -> Int -> NTree? -> Ntree
        Create an n-tree, which stores points in buckets of variable density.
        """
        self.split_dimension = split_dimension
        self.capacity = capacity

        self.points = points
        self.ranges = ranges
        self.dimensions = len(self.ranges)

        self.first_child, second_child = None, None
        self.has_children = False

        self.parent = parent

    def add_points(self, new_points, check_within_bounds=False):
        """
        [[Float]] -> Bool? -> ()
        Sort a batch of points into the n-tree, directing them to the relevant child if
        it is a branch.  Includes an optional toggle which first checks that the points
        fall within the n-tree (this is disabled for recusrive calls of this function.)
        """
        points, _ = (
            (new_points, None)
            if not check_within_bounds
            else self._filter(new_points, self.within_bounds),
        )

        if self.has_children:
            first_points, second_points = self._sort_by_child(points)
            self.first_child.add_points(first_points)
            self.second_child.add_points(second_points)
        else:
            self.points += points
            if self.point_count > self.capacity:
                self._create_children()

    def _sort_by_child(self, points):
        """
        [[Float]] -> ([[Float]], [[Float]])
        Sort a list of points into lists of whether they will be contained
        in the first or second child.
        """
        return self._filter(points, lambda p: p[self.dimension] < self.split_point)

    def split_point(self):
        """
        () -> Float
        Return the point within the split dimension at which a division is placed
        between the first and second child.
        """
        return 0.5 * sum(self.ranges[self.split_dimension])

    def within_bounds(self, point):
        """
        [[Float]] -> Bool
        Determine whether a point is contained within the bounds of the n-tree or
        its children.
        """
        return all([l <= x < u for x, (l, u) in zip(point, self.ranges)])

    def _create_children(self):
        """
        () -> ()
        Split the n-tree in half along its designated axis, creating two
        children and populating each of them with the points that fall into their
        respective bounds.
        """
        split_min, split_max = self.ranges[self.split_dimension]
        first_points, second_points = self._sort_by_child(self.points)
        split_dimension = (
            self.split_dimension + 1
            if self.split_dimension + 1 < self.dimensions
            else 0
        )

        self.has_children = True
        self.first_child = NTree(
            self.dimensions,
            split_dimension,
            self._override_list_value(
                self.ranges, self.split_dimension, (split_min, self.split_point)
            ),
            first_points,
            self.capacity,
            self,
        )
        self.second_child = NTree(
            self.dimensions,
            split_dimension,
            self._override_list_value(
                self.ranges, self.split_dimension, (self.split_point, split_max)
            ),
            second_points,
            self.capacity,
            self,
        )
        self.points = None

    @staticmethod
    def _override_list_value(values, index, new_value):
        """
        [a] -> Int -> a -> [a]
        Create a copy of a list with one value changed.  Supports wrapping.
        """
        i = index if index >= 0 else len(values) + index
        return values[:i] + [new_value] + values[i + 1 :]

    @staticmethod
    def _filter(values, predicate):
        """
        [a] -> (a -> Bool) -> ([a], [a])
        Filter a list of values by a predicate.
        """
        trues, falses = [], []
        for value in values:
            if predicate(value):
                trues.append(value)
            else:
                falses.append(value)
        return trues, falses

    @property
    def all_points(self):
        """
        () -> [[Float]]
        Return a list of all points contained either within the n-tree
        or its children.
        """
        return (
            self.points
            if not self.has_children
            else self.first_child.all_points + self.second_child.all_points
        )

    @property
    def point_count(self):
        """
        () -> Int
        Return the number of points in the n-tree.
        """
        return len(self.all_points)
