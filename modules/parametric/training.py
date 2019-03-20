import tensorflow as tf


class ParametricGeneratorTrainer:

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
            -> TrainingParameters -> ParametricGeneratorTrainer
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

    def load_data(self):
        """
        () -> (np.array, np.array, np.array)
        Load data from the dataset, returning it either as three numpy arrays or
        loading it from a serialised location and returning it that format.
        """
        raise NotImplementedError()


class TrainingParameters:

    def __init__(
        self,
        epochs,
        steps_per_epoch,
        batch_size=None,
        evaluation_sample_size=256,
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
            "epochs" = self.epochs,
            "stepsPerEpoch" = self.steps_per_epoch,
            "evaluationSampleSize" = self.evaluation_sample_size,
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
            json["evaluationSampleSize"] if "evaluationSampleSize" in json,
        )


def optimiser(loss, name="unnamed_optimiser"):
    """
    tf.Node -> String? -> tf.Op
    Create a default Adam optimiser for the given loss node.
    """
    return tf.train.AdamOptimizer(name=name).minimize(loss)
