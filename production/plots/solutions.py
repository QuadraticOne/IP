def get_solution_statistics(solution):
    """
    Dict -> (Float, Float, String)
    Extract satisfaction probability and relative density, as well as the solution
    type (true or generated) as a tuple from a dictionary describing them.
    """
    return (
        solution["satisfactionProbability"],
        solution["relativeDensity"],
        solution["type"],
    )
