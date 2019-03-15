class NTree:
    def __init__(
        self,
        ranges,
        capacity,
        split_dimension=0,
        points=[],
        minimum_width=0.001,
        parent=None,
    ):
        """
        [(Float, Float)] -> Int -> Int? -> [[Float]]? -> Float? -> NTree? -> NTree
        Create an n-tree, which stores points in buckets of variable density.
        """
        self.ranges = ranges
        self.dimensions = len(self.ranges)
        self.split_dimension = split_dimension

        self.capacity = capacity
        self.minimum_width = minimum_width

        self.points = points

        self.first_child, second_child = None, None
        self.has_children = False

        self.parent = parent
        self.check_for_overcrowding()

    def add_point(self, new_point, check_within_bounds=False):
        """
        [Float] -> Bool? -> ()
        Add a new point to the n-tree, sorting it into the relevant bucket.  An
        optional flag can be set to check that the point is actually within the
        bounds of the tree's bucket before adding it.
        """
        self.add_points([new_point], check_within_bounds=check_within_bounds)

    def add_points(self, new_points, check_within_bounds=False):
        """
        [[Float]] -> Bool? -> ()
        Sort a batch of points into the n-tree, directing them to the relevant child if
        it is a branch.  Includes an optional toggle which first checks that the points
        fall within the n-tree (this is disabled for recusrive calls of this function.)
        """
        if check_within_bounds:
            points, _ = self._partition(new_points, self.within_bounds)
        else:
            points = new_points

        if self.has_children:
            first_points, second_points = self._sort_by_child(points)
            self.first_child.add_points(first_points)
            self.second_child.add_points(second_points)
        else:
            self.points += points
            self.check_for_overcrowding()

    def check_for_overcrowding(self):
        """
        () -> ()
        Check that the capacity of the bucket has not been exceeded; if it has,
        split it into two smaller child buckets and redistribute the points to
        them.
        """
        can_subdivide = (
            self.ranges[self.split_dimension][1]
            > self.ranges[self.split_dimension][0] + self.minimum_width
        )
        if self.point_count > self.capacity and can_subdivide:
            self._create_children()

    def _sort_by_child(self, points):
        """
        [[Float]] -> ([[Float]], [[Float]])
        Sort a list of points into lists of whether they will be contained
        in the first or second child.
        """
        return self._partition(
            points, lambda p: p[self.split_dimension] < self.split_point
        )

    @property
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
            self._override_list_value(
                self.ranges, self.split_dimension, (split_min, self.split_point)
            ),
            self.capacity,
            split_dimension=split_dimension,
            points=first_points,
            minimum_width=self.minimum_width,
            parent=self,
        )
        self.second_child = NTree(
            self._override_list_value(
                self.ranges, self.split_dimension, (self.split_point, split_max)
            ),
            self.capacity,
            split_dimension=split_dimension,
            points=second_points,
            minimum_width=self.minimum_width,
            parent=self,
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
    def _partition(values, predicate):
        """
        [a] -> (a -> Bool) -> ([a], [a])
        Parition a list of values by a predicate.
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

    def __str__(self):
        """
        () -> String
        Return a string representation of the n-tree based on the number
        of points it contains.
        """
        return (
            str(self.point_count)
            if not self.has_children
            else "({}, {})".format(str(self.first_child), str(self.second_child))
        )
