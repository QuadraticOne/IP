import numpy as np


class EvaluationParameters:
    def __init__(
        self, constraint_samples, solutions_per_constraint_sample, satisfaction_cutoffs
    ):
        """
        Int -> Int -> [Float] -> EvaluationParameters
        Data class for storing parameters related to the evaluation JSON
        that should be produced after the experiment has run.
        """
        self.constraint_samples = constraint_samples
        self.solutions_per_constraint_sample = solutions_per_constraint_sample
        self.satisfaction_cutoffs = satisfaction_cutoffs

    def to_json(self):
        """
        () -> Dict
        Return a representation of a set of evaluation parameters as a JSON-like object.
        """
        return {
            "constraintSamples": self.constraint_samples,
            "solutionsPerConstraintSample": self.solutions_per_constraint_sample,
            "satisfactionCutoffs": self.satisfaction_cutoffs,
        }

    @staticmethod
    def from_json(json):
        """
        Dict -> EvaluationParameters
        Extract a set of evaluation parameters from a dictionary representation.
        """
        return EvaluationParameters(
            json["constraintSamples"],
            json["solutionsPerConstraintSample"],
            json["satisfactionCutoffs"],
        )

    def evaluate(self, trainer):
        """
        Metrics -> Dict
        Evaluate a set of trained networks and record their performance, according
        to the evaluation parameters laid out in the data class.
        """
        export = trainer.export()
        constraint_samples = self.make_constraint_samples(trainer)

    def _make_constraint_samples(self, trainer):
        """
        Trainer -> [np.array]
        Generate a list of constraints for which solutions are to be sampled.  If
        the evaluation parameters object's `constraint_samples` parameter is a list,
        this list will be returned.  Otherwise, its value is expected to be a string.
        In this case, it should be the type of sampling (currently only "uniform" is
        supported), followed by the number of constraints to sample, separated by
        a comma, such as "uniform, 64".
        """
        if isinstance(self.constraint_samples, list) or isinstance(
            self.constraint_samples, np.ndarray
        ):
            return self.constraint_samples
        else:
            if not isinstance(self.constraint_samples, str):
                raise ValueError(
                    "constraint_samples should be a list of numpy arrays or a string"
                )

            args = self.constraint_samples.replace(" ", "").split(",")
            n = 64 if len(args) < 2 else int(args[1])
            sampling_method = args[0]

            if sampling_method == "unform":
                g = trainer.parametric_generator
                return np.random.uniform(
                    low=-1.0, high=1.0, size=(n, g.constraint_dimension)
                )

            else:
                raise ValueError("unknown sampling method '{}'".format(sampling_method))

