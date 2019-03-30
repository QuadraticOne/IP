from environments.environment import VectorEnvironment, UniformVectorEnvironment


circles = UniformVectorEnvironment(
    3,
    3,
    lambda scp: (scp[1][0] - scp[0][0]) ** 2 + (scp[1][1] - scp[0][1]) ** 2
    > (scp[1][2] - scp[0][2]) ** 2,
    solution_range=(-1.0, 1.0),
    constraint_range=(-1.0, 1.0),
)


def make_circles_datasets(sizes=[64, 128, 256, 1024]):
    """
    [Int] -> ()
    Create a number of datasets for the circles environment under
    production/datasets/circles.
    """
    for size in sizes:
        circles.save_dataset(size, "production/datasets/circles/", str(size))
