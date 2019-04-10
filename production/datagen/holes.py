from environments.environment import VectorEnvironment, UniformVectorEnvironment
from modules.parametric.generator import ParametricGenerator
from modules.parametric.trainer import Trainer
import tensorflow as tf


holes = UniformVectorEnvironment(
    3,
    3,
    lambda scp: (scp[1][0] - scp[0][0]) ** 2 + (scp[1][1] - scp[0][1]) ** 2
    > (scp[1][2] - scp[0][2]) ** 2,
    solution_range=(-1.0, 1.0),
    constraint_range=(-1.0, 1.0),
)


def make_holes_datasets(sizes=[64, 128, 256, 1024]):
    """
    [Int] -> ()
    Create a number of datasets for the holes environment under
    production/datasets/holes.
    """
    for size in sizes:
        holes.save_dataset(size, "production/datasets/holes/", str(size))


def default_holes_trainer(dataset_size):
    """
    String -> Trainer
    Create a default trainer for the holes environment which can be
    altered before running experiments.
    """
    generator = ParametricGenerator("generator", 3, 3, 3, 2)
    generator.build_input_nodes()

    layer = (4, "leaky-relu")
    generator.set_discriminator_architecture([layer])
    generator.set_embedder_architecture(
        [layer, layer], "leaky-relu", [layer, layer], "leaky-relu"
    )
    generator.set_generator_architecture([layer, layer], "leaky-relu", "tanh")

    return Trainer(
        generator,
        "production/datasets/holes/{}".format(str(dataset_size)),
        1.0,
        Trainer.TrainingParameters(20, 20000, batch_size=32),
        Trainer.TrainingParameters(20, 20000, batch_size=32),
        Trainer.TrainingParameters(20, 20000, batch_size=32),
        evaluation_parameters=Trainer.EvaluationParameters(
            {"quantity": 1, "samplingMethod": "uniform"}, 1, 1, 1024, 64, 16, 2048
        ),
        experiment_log_folder="production/datasets/holes/",
    )


def generate_holes_data(dataset_size, recall_weight=None, log=None):
    """
    Int -> Float? -> Bool? -> ()
    Train five discriminator/generator pairs on the holes dataset of the given
    size and log the results under production/datasets/holes/results-<size>.
    """
    trainer = default_holes_trainer(dataset_size)
    if recall_weight is not None:
        trainer.recall_weight = recall_weight
    if log is not None:
        trainer.log = log

    trainer.log_experiments(
        "results-{}".format(dataset_size), 3, reset=trainer.reset_training
    )
