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

    def evaluate(self, metrics):
        """
        Metrics -> Dict
        Evaluate a set of trained networks and record their performance, according
        to the evaluation parameters laid out in the data class.
        """
        return {"samples": self.take_sample(metrics)}

    def take_sample(self, metrics, constraint=None):
        """
        Metrics -> np.array? -> Dict
        Take a sample from the network, optionally putting in a constraint and taking
        a sample uniformly from the latent space.
        """
        outputs = [
            metrics.generator["output"],
            metrics.discriminator["output"],
            metrics.generator["input"],
        ]
        solution, satisfaction, latent = (
            metrics.trainer.session.run(
                outputs, feed_dict={metrics.embedder["input"]: constraint}
            )
            if constraint is not None
            else metrics.trainer.session.run(outputs)
        )
        return {
            "constraint": constraint,
            "solution": solution,
            "satisfaction": satisfaction,
            "latent": latent,
        }
