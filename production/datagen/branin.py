from environments.environment import UniformVectorEnvironment
from modules.parametric.generator import ParametricGenerator
from modules.parametric.trainer import Trainer
from math import pi, cos
import tensorflow as tf


branin = UniformVectorEnvironment(
    2,
    2,
    lambda scp: scp[1][0] * scp[1][1]
    <= branin_function(scp[0][0], scp[0][1])
    <= scp[1][1],
    solution_range=(0.0, 1.0),
    constraint_range=(0.0, 1.0),
)


def branin_function(x, y):
    """
    Float -> Float -> Float
    Python implementation of the Branin function for testing optimisation
    algorithms.
    """
    x_scaled = (x * 15) - 5
    y_scaled = y * 15
    a = 1.0
    b = 5.1 / (4 * pi * pi)
    c = 5 / pi
    r = 6.0
    s = 10.0
    t = 1 / (8 * pi)
    lhs = a * (y_scaled - b * x_scaled * x_scaled + c * x_scaled - r) ** 2
    rhs = s * (1 - t) * cos(x_scaled) + s
    return (lhs + rhs) / (250 * 1.24)


def make_branin_datasets(sizes=[64, 256, 1024]):
    """
    [Int] -> ()
    Create a number of datasets for the Branin function environment under
    production/datasets/holes.
    """
    for size in sizes:
        branin.save_dataset(size, "production/datasets/branin/", str(size))


def default_branin_trainer(dataset_size):
    """
    String -> Trainer
    Create a default trainer for the Branin function environment which can be
    altered before running experiments.
    """
    generator = ParametricGenerator("generator", 2, 2, 2, 8)

    generator.solution_lower_bound, generator.solution_upper_bound = 0.0, 1.0
    generator.constraint_lower_bound, generator.constraint_upper_bound = 0.0, 1.0

    generator.build_input_nodes()

    layer = (32, "leaky-relu")
    generator.set_discriminator_architecture([(64, "leaky-relu"), (64, "leaky-relu")])
    generator.set_embedder_architecture(
        [layer, layer], "leaky-relu", [layer, layer], "leaky-relu"
    )
    generator.set_generator_architecture([layer, layer], "leaky-relu", "sigmoid")

    return Trainer(
        generator,
        "production/datasets/branin/{}".format(str(dataset_size)),
        1.0,
        Trainer.TrainingParameters(64, 20000, batch_size=32),
        Trainer.TrainingParameters(16, 20000, batch_size=32),
        Trainer.TrainingParameters(128, 20000, batch_size=32),
        evaluation_parameters=Trainer.EvaluationParameters(
            {"quantity": 16, "samplingMethod": "uniform"}, 128, 128, 1024, 64, 128, 8192
        ),
        experiment_log_folder="production/datasets/branin/",
    )


def generate_branin_data(
    dataset_size,
    recall_weight=None,
    log=None,
    recall_proxy=None,
    recall_proxy_metadata=None,
):
    """
    Int -> Float? -> Bool? -> String? -> Dict? -> ()
    Train five discriminator/generator pairs on the holes dataset of the given
    size and log the results under production/datasets/holes/results-<size>.
    """
    trainer = default_branin_trainer(dataset_size)
    if recall_weight is not None:
        trainer.recall_weight = recall_weight
    if log is not None:
        trainer.log = log
    if recall_proxy is not None:
        trainer.pretraining_loss = recall_proxy
        trainer.recall_proxy = recall_proxy
    if recall_proxy_metadata is not None:
        trainer.pretraining_loss_metadata = recall_proxy_metadata
        trainer.recall_proxy_metadata = recall_proxy_metadata

    trainer.log_experiments(
        "results-{}".format(dataset_size), 5, reset=trainer.reset_training
    )
