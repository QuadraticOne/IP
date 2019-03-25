from modules.parametric.generator import ParametricGenerator
from modules.parametric.metrics import Metrics
from wise.util.io import IO
from wise.training.routines import fit
from wise.training.experiments.experiment import Experiment
import wise.util.training as tu
import modules.sampling as sample
import modules.parametric.evaluation as evaluation
import tensorflow as tf
import numpy as np
import modules.reusablenet as rnet
import time


class Trainer(Experiment):

    EvaluationParameters = evaluation.EvaluationParameters

    def __init__(
        self,
        parametric_generator,
        dataset,
        recall_weight,
        discriminator_training_parameters,
        generator_pretraining_parameters,
        generator_training_parameters,
    ):
        """
        ParametricGenerator -> Either String [([Float], [Float], Bool)]
            -> Float -> TrainingParameters -> TrainingParameters
            -> TrainingParameters -> Trainer
        Create a class for training parametric generators.
        """
        self.parametric_generator = parametric_generator
        self.dataset = dataset
        self.recall_weight = recall_weight
        self.discriminator_training_parameters = discriminator_training_parameters
        self.generator_pretraining_parameters = generator_pretraining_parameters
        self.generator_training_parameters = generator_training_parameters

        self.solutions, self.constraints, self.satisfactions = self.load_data()

        self.parametric_generator.solution_dimension = self.solutions.shape[1]
        self.parametric_generator.constraint_dimension = self.constraints.shape[1]

        self.generator_pretraining_parameters.batch_size = (
            parametric_generator.generator_training_batch_size
        )
        self.generator_training_parameters.batch_size = (
            parametric_generator.generator_training_batch_size
        )

        self.session = None

        self.validation_proportion = 0.2
        self.pretraining_loss = "uniformity"
        self.precision_proxy = "precision"
        self.recall_proxy = "uniformity"

        self.log = False

        self.set_discriminator_inputs()

    def load_data(self):
        """
        () -> (np.array, np.array, np.array)
        Load data from the dataset, returning it either as three numpy arrays or
        loading it from a serialised location and returning it that format.
        """
        data = None

        if not isinstance(self.dataset, str):
            data = self.dataset
        else:
            data = IO([p + "/" for p in self.dataset.split("/")[:-1]]).restore_object(
                self.dataset.split(".")[-1]
            )

        true, false = 1.0, 0.0
        solutions, constraints, satisfactions = [], [], []
        for datum in data:
            solutions.append(np.array(datum[0]))
            constraints.append(np.array(datum[1]))
            satisfactions.append(true if datum[2] else false)

        solution_dimension = np.array(solutions).shape[1]
        if solution_dimension != self.parametric_generator.solution_dimension:
            raise Exception(
                "solution dimension expected to be {} but was {}".format(
                    self.parametric_generator.solution_dimension, solution_dimension
                )
            )
        constraint_dimension = np.array(constraints).shape[1]
        if constraint_dimension != self.parametric_generator.constraint_dimension:
            raise Exception(
                "constraint dimension expected to be {} but was {}".format(
                    self.parametric_generator.constraint_dimension, constraint_dimension
                )
            )

        return np.array(solutions), np.array(constraints), np.array(satisfactions)

    def reset_training(self):
        """
        () -> ()
        Reset all current progress, returning any weight matrices to random values.
        """
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def set_discriminator_inputs(self):
        """
        () -> ()
        Create input nodes for solutions, constraints, and satisfaction labels, for
        both training and validation.
        """
        make_nodes = sample.training_input_nodes(
            len(self.solutions),
            self.discriminator_training_parameters.batch_size,
            validation_proportion=self.validation_proportion,
        )
        self.solution_sample, self.solution_validation = make_nodes(self.solutions)
        self.constraint_sample, self.constraint_validation = make_nodes(
            self.constraints
        )
        self.satisfaction_sample, self.satisfaction_validation = make_nodes(
            self.satisfactions
        )

    def train_discriminator(self):
        """
        () -> Dict
        Train the discriminator, assuming it has just been reset and that samples
        are balanced to ensure an equal proportion of each label, and return data
        about the training progress contained in a JSON-like object.
        """
        if self.log:
            print("\nDiscriminator training:")

        data = {}

        training_discriminator = self.parametric_generator.build_discriminator(
            self.solution_sample, self.constraint_sample
        )
        target, loss, accuracy, optimiser, init = tu.classification_metrics_with_initialiser(
            [],
            training_discriminator["output"],
            "training_discriminator_nodes",
            variables=rnet.all_variables(training_discriminator),
            target=self.satisfaction_sample,
        )
        self.session.run(init)
        metrics = [("Loss", loss), ("Accuracy", accuracy)] if self.log else None

        data["before"] = {
            "loss": self.session.run(loss),
            "accuracy": self.session.run(accuracy),
        }

        data["startTime"] = time.time()
        self.discriminator_training_parameters.fit(
            self.session, optimiser, metrics=metrics
        )
        data["endTime"] = time.time()
        data["duration"] = data["endTime"] - data["startTime"]

        data["after"] = {
            "trainingLoss": self.session.run(loss),
            "trainingAccuracy": self.session.run(accuracy),
        }

        validation_discriminator = self.parametric_generator.build_discriminator(
            self.solution_validation, self.constraint_validation
        )
        target, loss, accuracy, _ = tu.classification_metrics(
            [],
            validation_discriminator["output"],
            "validation_discriminator_nodes",
            variables=rnet.all_variables(validation_discriminator),
            target=self.satisfaction_validation,
        )

        data["after"]["validationLoss"], data["after"][
            "validationAccuracy"
        ] = self.session.run([loss, accuracy])

        return data

    def pretrain_generator(self):
        """
        () -> Dict
        Pretrain the generator, as well as the constraint embedding subnetwork.
        """
        if self.log:
            print("\nGenerator pretraining:")

        data = {}

        weights, biases = self.parametric_generator.build_embedder(
            self.parametric_generator.make_constraint_sample_node()
        )
        generator = self.parametric_generator.build_generator(
            self.parametric_generator.make_latent_sample_node(), weights, biases
        )

        loss = self.metrics(generator=generator).get(self.pretraining_loss)
        optimiser, init = tu.default_adam_optimiser_with_initialiser(
            loss,
            "pretraining_optimiser",
            variables=rnet.all_variables([weights, biases, generator]),
        )
        self.session.run(init)

        data["before"] = {"loss": self.session.run(loss)}
        metrics = [("Loss", loss)] if self.log else None

        data["startTime"] = time.time()

        self.generator_pretraining_parameters.fit(
            self.session, optimiser, metrics=metrics
        )

        data["endTime"] = time.time()
        data["duration"] = data["endTime"] - data["startTime"]

        data["after"] = {"loss": self.session.run(loss)}

        return data

    def train_generator(self):
        """
        () -> Dict
        Train the generator and constraint embedding subnetwork, returning various
        metrics and results in a JSON-like object.
        """
        if self.log:
            print("\nGenerator training:")

        data = {}

        constraint_input = self.parametric_generator.make_constraint_sample_node()
        weights, biases = self.parametric_generator.build_embedder(constraint_input)
        generator = self.parametric_generator.build_generator(
            self.parametric_generator.make_latent_sample_node(), weights, biases
        )
        discriminator = self.parametric_generator.build_discriminator(
            generator["output"], constraint_input
        )

        metrics = self.metrics(generator=generator, discriminator=discriminator)

        precision_proxy = metrics.get(self.precision_proxy)
        recall_proxy = metrics.get(self.recall_proxy)
        loss = precision_proxy + self.recall_weight * recall_proxy

        optimiser, init = tu.default_adam_optimiser_with_initialiser(
            loss,
            "generator_training_loss",
            rnet.all_variables([weights, biases, generator, discriminator]),
        )

        logging_metrics = (
            [
                ("Loss", loss),
                ("Precision proxy", precision_proxy),
                ("Recall proxy", recall_proxy),
            ]
            if self.log
            else None
        )

        self.session.run(init)

        data["before"] = {}
        data["before"]["loss"], data["before"]["precisionProxy"], data["before"][
            "recallProxy"
        ] = self.session.run([loss, precision_proxy, recall_proxy])

        data["startTime"] = time.time()

        self.generator_training_parameters.fit(
            self.session, optimiser, metrics=logging_metrics
        )

        data["endTime"] = time.time()
        data["duration"] = data["endTime"] - data["startTime"]

        data["after"] = {}
        data["after"]["loss"], data["after"]["precisionProxy"], data["after"][
            "recallProxy"
        ] = self.session.run([loss, precision_proxy, recall_proxy])

        return data

    def run_experiment(self, log=None):
        """
        Bool? -> Dict
        Reset the training and run it from start to finish, returning the results
        within a JSON-like object.
        """
        if log is not None:
            old_log = self.log
            self.log = log

        self.reset_training()

        data = {"parameters": self.to_json()}
        data["discriminatorTraining"] = self.train_discriminator()
        data["generatorPretraining"] = self.pretrain_generator()
        data["generatorTraining"] = self.train_generator()

        if log is not None:
            self.log = old_log
        return data

    def to_json(self):
        """
        () -> Dict
        Create a JSON-like representation of the generator's training parameters.
        """
        return {
            "parametricGenerator": self.parametric_generator.to_json(),
            "dataset": self.dataset if isinstance(self.dataset, str) else "literal",
            "recallWeight": self.recall_weight,
            "discriminatorTrainingParameters": (
                self.discriminator_training_parameters.to_json()
            ),
            "generatorPretrainingParameters": (
                self.generator_pretraining_parameters.to_json()
            ),
            "generatorTrainingParameters": (
                self.generator_training_parameters.to_json()
            ),
            "discriminatorValidationProportion": self.validation_proportion,
            "pretrainingLoss": self.pretraining_loss,
            "precisionProxy": self.precision_proxy,
            "recallProxy": self.recall_proxy,
        }

    @staticmethod
    def from_json(json):
        """
        Dict -> Trainer
        Create a trainer for a parametric generator from a serialised list of
        parameters.
        """

        def training_pars(key):
            return Trainer.TrainingParameters.from_json(json[key])

        trainer = Trainer(
            ParametricGenerator.from_json(json["parametricGenerator"]),
            json["dataset"] if "dataset" in json else [],
            json["recallWeight"],
            training_pars("discriminatorTrainingParameters"),
            training_pars("generatorPretrainingParameters"),
            training_pars("generatorTrainingParameters"),
        )

        if "discriminatorValidationProportion" in json:
            trainer.validation_proportion = json["discriminatorValidationProportion"]
        if "pretrainingLoss" in json:
            trainer.pretraining_loss = json["pretrainingLoss"]
        if "precisionProxy" in json:
            trainer.precision_proxy = json["precisionProxy"]
        if "recallProxy" in json:
            trainer.recall_proxy = json["recallProxy"]

        return trainer

    def metrics(self, generator=None, embedder=None, discriminator=None):
        """
        () -> Metrics
        Return an object that can be used to quickly retrieve various metrics
        useful in the training of the generator.
        """
        return Metrics(
            self, generator=generator, embedder=embedder, discriminator=discriminator
        )

    class TrainingParameters:
        def __init__(
            self, epochs, steps_per_epoch, batch_size=None, evaluation_sample_size=256
        ):
            """
            Int -> Int -> Int? -> Int? -> TrainingParameters
            Data class for parameterising a training pass on a neural network.
            """
            self.epochs = epochs
            self.steps_per_epoch = steps_per_epoch
            self.batch_size = batch_size
            self.evaluation_sample_size = evaluation_sample_size

        def to_json(self):
            """
            () -> Dict
            Create a JSON-like representation of the training parameters.
            """
            json = {
                "epochs": self.epochs,
                "stepsPerEpoch": self.steps_per_epoch,
                "evaluationSampleSize": self.evaluation_sample_size,
            }
            if self.batch_size is not None:
                json["batchSize"] = self.batch_size
            return json

        @staticmethod
        def from_json(json):
            """
            Dict -> TrainingParameters
            Create a set of training parameters from a suitable JSON-like object.
            """
            return TrainingParameters(
                json["epochs"],
                json["stepsPerEpoch"],
                json["batchSize"] if "batchSize" in json else None,
                json["evaluationSampleSize"],
            )

        def fit(self, session, optimise_op, metrics=None, sampler=None):
            """
            tf.Session -> tf.Op -> [(String, tf.Node)]? -> Sampler? -> ()
            Fit a model using the parameters defined within this data object.
            """
            fit(
                session,
                optimise_op,
                sampler,
                self.epochs,
                self.steps_per_epoch,
                self.batch_size,
                metrics,
                evaluation_sample_size=self.evaluation_sample_size,
            )


def optimiser(loss, name="unnamed_optimiser"):
    """
    tf.Node -> String? -> tf.Op
    Create a default Adam optimiser for the given loss node.
    """
    return tf.train.AdamOptimizer(name=name).minimize(loss)
