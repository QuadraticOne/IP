from maths.ntree import NTree
from maths.mcmc import mcmc_samples
import numpy as np


class EvaluationParameters:
    def __init__(
        self,
        constraint_samples,
        generated_solutions_per_constraint,
        true_solutions_per_constraint,
        monte_carlo_burn_in,
        monte_carlo_sample_gap,
        n_tree_bucket_size,
        n_tree_population,
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
        self.monte_carlo_sample_gap = monte_carlo_sample_gap

        self.n_tree_bucket_size = n_tree_bucket_size
        self.n_tree_population = n_tree_population

    def to_json(self):
        """
        () -> Dict
        Return a representation of a set of evaluation parameters as a JSON-like object.
        """
        return {
            "constraintSamples": self.constraint_samples,
            "generatedSolutionsPerConstraint": self.generated_solutions_per_constraint,
            "trueSolutionsPerConstraint": self.true_solutions_per_constraint,
            "monteCarlo": {
                "burnIn": self.monte_carlo_burn_in,
                "sampleGap": self.monte_carlo_sample_gap,
            },
            "nTree": {
                "bucketSize": self.n_tree_bucket_size,
                "population": self.n_tree_population,
            },
        }

    @staticmethod
    def from_json(json):
        """
        Dict -> EvaluationParameters
        Extract a set of evaluation parameters from a dictionary representation.
        """
        return EvaluationParameters(
            json["constraintSamples"],
            json["generatedSolutionsPerConstraint"],
            json["trueSolutionsPerConstraint"],
            json["monteCarlo"]["burnIn"],
            json["monteCarlo"]["sampleGap"],
            json["nTree"]["bucketSize"],
            json["nTree"]["population"],
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
            {"constraint": constraint, "solutions": []}
            for constraint in constraint_samples
        ]

        if trainer.log:
            i = 0
            print()
        for constraint in data["constraintSamples"]:
            if trainer.log:
                i += 1
                print(
                    "Evaluating constraint {}/{}...".format(
                        i, len(data["constraintSamples"])
                    )
                )

            self.append_generated_solutions(constraint, export)
            self.append_true_solutions(constraint, export, trainer)
            self.calculate_solution_properties(constraint, export, trainer)
            constraint["summary"] = self.calculate_solution_statistics(constraint)

        data["summary"] = self.calculate_solution_statistics(data["constraintSamples"])

        if trainer.log:
            print("Done.")

        return data

    def append_generated_solutions(self, constraint, export):
        """
        Dict -> ExportedParametricGenerator -> ()
        Append `generated_solutions_per_constraint_sample` solutions, taken
        from the learned generator, to the list of solutions for each constraint.
        """
        solutions = export.sample_for_constraint(
            constraint["constraint"], self.generated_solutions_per_constraint
        )
        for solution in solutions:
            constraint["solutions"].append(
                {
                    "solution": solution.solution,
                    "latent": solution.latent,
                    "type": "generated",
                }
            )

    def append_true_solutions(self, constraint, export, trainer):
        """
        Dict -> ExportedParametricGenerator -> Trainer -> ()
        Append `true_solutions_per_constraint` solutions, taken from the true target
        distribution using a Markov chain, to the list of solutions for each constraint.
        """
        f = export.satisfaction_probability(constraint["constraint"])
        solutions = mcmc_samples(
            f,
            self.true_solutions_per_constraint,
            self.monte_carlo_sample_gap,
            self.monte_carlo_burn_in,
            [0.5] * trainer.parametric_generator.solution_dimension,
        )[1:]
        for solution in solutions:
            constraint["solutions"].append({"solution": solution, "type": "true"})

    def calculate_solution_properties(self, constraint, export, trainer):
        """
        Dict -> ExportedParametricGenerator -> Trainer -> ()
        Calculate satisfaction probability and relative density in the latent
        space of each solution to a constraint.
        """
        satisfaction_probability = export.satisfaction_probability(
            constraint["constraint"]
        )

        ntree = self._make_n_tree(export, constraint["constraint"], trainer)
        for solution in constraint["solutions"]:
            solution["relativeDensity"] = ntree.relative_density_at_point(
                solution["solution"], error_if_out_of_bounds=False
            )
            solution["satisfactionProbability"] = satisfaction_probability(
                solution["solution"]
            )

    def calculate_solution_statistics(self, constraint):
        """
        Either [Dict] Dict -> Dict
        Calculate summary statistics for each solution to a constraint or set
        of constraints.
        """
        return {
            "all": self.summarise_relative_density_and_satisfaction_probability(
                self.solutions_of_type(constraint)
            ),
            "true": self.summarise_relative_density_and_satisfaction_probability(
                self.solutions_of_type(constraint, "true")
            ),
            "generated": self.summarise_relative_density_and_satisfaction_probability(
                self.solutions_of_type(constraint, "generated")
            ),
        }

    def solutions_of_type(self, constraint, type_name=None):
        """
        Either Dict [Dict] -> String? -> [Dict]
        Return all the solutions presented for either a single constraint or
        a list of constraints whose type matches the given type name.  If no type
        name is specified then all solutions will be returned.
        """
        return _flatmap(
            lambda c: [
                solution
                for solution in c["solutions"]
                if type_name is None or solution["type"] == type_name
            ],
            _list_wrap(constraint),
        )

    def summarise_relative_density_and_satisfaction_probability(self, solutions):
        """
        [Dict] -> Dict
        Iterate through a list of solutions and extract statistics on the maximum,
        minimum, and mean relative densities and satisfaction probabilities.
        """
        return {
            "relativeDensity": self.summarise_values(
                [s["relativeDensity"] for s in solutions], ""
            ),
            "satisfactionProbability": self.summarise_values(
                [s["satisfactionProbability"] for s in solutions], ""
            ),
        }

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

    def _make_n_tree(self, export, constraint, trainer):
        """
        ExportedParametricGenerator -> np.array -> Trainer -> NTree
        Build a suitably sized n-tree and populate it with samples from the
        generator.
        """
        ntree = NTree(
            [
                (
                    trainer.parametric_generator.solution_lower_bound,
                    trainer.parametric_generator.solution_upper_bound,
                )
            ]
            * trainer.parametric_generator.solution_dimension,
            self.n_tree_bucket_size,
        )

        samples = export.sample_for_constraint(constraint, self.n_tree_population)
        ntree.add_points([sample.solution for sample in samples])
        return ntree

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
