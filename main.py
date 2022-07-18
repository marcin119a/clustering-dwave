from tools import *
import math 
import dwavebinarycsp
import dwave.inspector
from dwave.system import EmbeddingComposite, DWaveSampler


def main(scattered_points, filename, problem_inspector):
    """Perform clustering analysis on given points

    Args:
        scattered_points (list of tuples):
            Points to be clustered
        filename (str):
            Output file for graphic
        problem_inspector (bool):
            Whether to show problem inspector
    """
    # Set up problem
    # Note: max_distance gets used in division later on. Hence, the max(.., 1)
    #   is used to prevent a division by zero
    coordinates = [Coordinate(x, y) for x, y in scattered_points]
    max_distance = max(get_max_distance(coordinates), 1)

    # Build constraints
    csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)

    # Apply constraint: coordinate can only be in one colour group
    choose_one_group = {(0, 0, 1), (0, 1, 0), (1, 0, 0)}
    for coord in coordinates:
        csp.add_constraint(choose_one_group, (coord.r, coord.g, coord.b))

    # Build initial BQM
    bqm = dwavebinarycsp.stitch(csp)

    # Edit BQM to bias for close together points to share the same color
    for i, coord0 in enumerate(coordinates[:-1]):
        for coord1 in coordinates[i+1:]:
            # Set up weight
            d = get_distance(coord0, coord1) / max_distance  # rescale distance
            weight = -math.cos(d*math.pi)

            # Apply weights to BQM
            bqm.add_interaction(coord0.r, coord1.r, weight)
            bqm.add_interaction(coord0.g, coord1.g, weight)
            bqm.add_interaction(coord0.b, coord1.b, weight)

    # Edit BQM to bias for far away points to have different colors
    for i, coord0 in enumerate(coordinates[:-1]):
        for coord1 in coordinates[i+1:]:
            # Set up weight
            # Note: rescaled and applied square root so that far off distances
            #   are all weighted approximately the same
            d = math.sqrt(get_distance(coord0, coord1) / max_distance)
            weight = -math.tanh(d) * 0.1

            # Apply weights to BQM
            bqm.add_interaction(coord0.r, coord1.b, weight)
            bqm.add_interaction(coord0.r, coord1.g, weight)
            bqm.add_interaction(coord0.b, coord1.r, weight)
            bqm.add_interaction(coord0.b, coord1.g, weight)
            bqm.add_interaction(coord0.g, coord1.r, weight)
            bqm.add_interaction(coord0.g, coord1.b, weight)

    # Submit problem to D-Wave sampler
    sampler = EmbeddingComposite(DWaveSampler(token='DEV-ad524c7e5d8452a5b2546ba4913ae5a169926032'))
    sampleset = sampler.sample(bqm,
                               chain_strength=4,
                               num_reads=1000,
                               label='Example - Clustering')
    best_sample = sampleset.first.sample

    # Visualize graph problem
    if problem_inspector:
        dwave.inspector.show(bqm, sampleset)

    # Visualize solution
    groupings = get_groupings(best_sample)
    visualize_groupings(groupings, filename)

    # Print solution onto terminal
    # Note: This is simply a more compact version of 'best_sample'
    print(groupings)


if __name__ == "__main__":
    # Simple, hardcoded data set
    scattered_points = [(0, 0), (1, 1), (2, 4), (3, 2)]

    # Save the original, un-clustered plot
    orig_filename = "four_points.png"
    visualize_scatterplot(scattered_points, orig_filename)

    # Find clusters
    # Note: the key part of this demo is in the construction of this function
    clustered_filename = "four_points_clustered.png"
    main(scattered_points, clustered_filename, False)

    print("Your plots are saved to '{}' and '{}'.".format(orig_filename,
                                                    clustered_filename))