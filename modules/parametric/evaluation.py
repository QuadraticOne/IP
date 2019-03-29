import numpy as np


class EvaluationParameters:
    def __init__(
        self,
        constraint_samples,
        generated_solutions_per_constraint,
        true_solutions_per_constraint,
        monte_carlo_burn_in,
        monte_carlo_samples,
        monte_carlo_sample_gap,
        n_tree_bucket_size,
    ):
        """
        Either String [np.array] -> Int -> Int -> Int -> Int -> Int -> Int
            -> EvaluationParameters
        Data class for storing parameters related to the evaluation JSON
        that should be produced after the experiment has run.
        """
        self.constraint_samples = constraint_samples
        self.generated_solutions_per_constraint = generated_solutions_per_constraint
        self.true_solutions_per_constraint = true_solutions_per_constraint

        self.monte_carlo_burn_in = monte_carlo_burn_in
        self.monte_carlo_samples = monte_carlo_samples
        self.monte_carlo_sample_gap = monte_carlo_sample_gap

        self.n_tree_bucket_size = n_tree_bucket_size

    def to_json(self):
        """
        () -> Dict
        Return a representation of a set of evaluation parameters as a JSON-like object.
        """
        return {
            "constraintSamples": self.constraint_samples,
            "solutionsPerConstraintSample": self.generated_solutions_per_constraint,
            "trueSolutionsPerConstraint": self.true_solutions_per_constraint,
            "monteCarlo": {
                "burnIn": self.monte_carlo_burn_in,
                "samples": self.monte_carlo_samples,
                "sampleGap": self.monte_carlo_sample_gap,
            },
            "nTreeBucketSize": self.n_tree_bucket_size,
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
            json["trueSolutionsPerConstraint"],
            json["monteCarlo"]["burnIn"],
            json["monteCarlo"]["samples"],
            json["monteCarlo"]["sampleGap"],
            json["nTreeBucketSize"],
        )

    def evaluate(self, trainer):
        """
        Metrics -> Dict
        Evaluate a set of trained networks and record their performance, according
        to the evaluation parameters laid out in the data class.
        """
        export = trainer.export()
        data = {}

        constraint_samples = self._make_constraint_samples(trainer)
        data["constraintSamples"] = [
            {
                "constraint": constraint.tolist(),
                "solutions": [
                    result.to_json()
                    for result in _list_wrap(
                        export.sample_for_constraint(
                            constraint, self.generated_solutions_per_constraint
                        )
                    )
                ],
            }
            for constraint in constraint_samples
        ]

        for constraint in data["constraintSamples"]:
            constraint["summary"] = self.summarise_values(
                [
                    solution["satisfactionProbability"]
                    for solution in constraint["solutions"]
                ],
                "Satisfaction",
            )

        data["constraintSamplesSummary"] = self.summarise_values(
            _flatmap(
                lambda c: [s["satisfactionProbability"] for s in c["solutions"]],
                data["constraintSamples"],
            ),
            "Satisfaction",
        )

        return data

    def summarise_values(self, values, name):
        """
        [Float] -> String -> Dict
        Summarise the values, producing statistics such as maximum, minimum, and mean.
        """
        return {
            "minimum" + name: min(values),
            "maximum" + name: max(values),
            "mean" + name: sum(values) / len(values),
        }

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

            if sampling_method == "uniform":
                g = trainer.parametric_generator
                return np.random.uniform(
                    low=-1.0, high=1.0, size=(n, g.constraint_dimension)
                )

            else:
                raise ValueError("unknown sampling method '{}'".format(sampling_method))


def _list_wrap(value):
    """
    Either a [a] -> [a]
    Wrap a value in a list if it is not already a list.
    """
    return value if isinstance(value, list) else [value]


def _flatmap(f, iterable):
    """
    (a -> [b]) -> [a] -> [b]
    Map over an iterable and concatenate the results into a single list.
    """
    output = []
    for i in iterable:
        output += f(i)
    return output