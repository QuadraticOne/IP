class NTree:
    def __init__(self, dimensions, split_dimension, ranges, children, points, capacity):
        """
        Int -> Int -> [(Float, Float)] -> (NTree?, NTree?) -> [[Float]] -> Int -> Ntree
        Create an n-tree, which stores points in buckets of variable density.
        """
        self.dimensions = dimensions
        self.split_dimension = split_dimension
        self.capacity = capacity

        self.points = points
        self.ranges = ranges

        self.first_child, second_child = None, None
        self.has_children = False

    def _create_children(self):
        """
        () -> ()
        Split the n-tree in half along its designated axis, creating two
        children and populating each of them with the points that fall into their
        respective bounds.
        """
        split_min, split_max = self.ranges[self.split_dimension]
        split_point = 0.5 * (split_min + split_max)
        first_points, second_points = self._filter(
            self.points, lambda p: p[self.split_dimension] < split_point
        )
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
                self.ranges, self.split_dimension, (split_min, split_point)
            ),
            first_points,
            self.capacity,
        )

        self.second_child = NTree(
            self.dimensions,
            split_dimension,
            self._override_list_value(
                self.ranges, self.split_dimension, (split_point, split_max)
            ),
            second_points,
            self.capacity,
        )

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
