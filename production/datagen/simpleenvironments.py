from environments.environment import VectorEnvironment, UniformVectorEnvironment
from modules.parametric.generator import ParametricGenerator
from modules.parametric.trainer import Trainer


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


def default_circles_trainer(dataset_size):
    """
    String -> Trainer
    Create a default trainer for the Circles environment which can be
    altered before running experiments.
    """
    generator = ParametricGenerator("generator", 3, 3, 3, 5)
    generator.build_input_nodes()

    layer = (4, "leaky-relu")
    generator.set_discriminator_architecture([layer])
    generator.set_embedder_architecture(
        [layer, layer], "leaky-relu", [layer, layer], "leaky-relu"
    )
    generator.set_generator_architecture([layer, layer], "leaky-relu", "tanh")

    return Trainer(
        generator,
        "production/datasets/circles/{}".format(str(dataset_size)),
        1.0,
        Trainer.TrainingParameters(20, 20000, batch_size=32),
        Trainer.TrainingParameters(20, 20000, batch_size=32),
        Trainer.TrainingParameters(20, 20000, batch_size=32),
        evaluation_parameters=Trainer.EvaluationParameters(
            {"quantity": 32, "samplingMethod": "uniform"}, 128, 128, 1024, 64, 16, 2048
        ),
    )
