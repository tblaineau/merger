
import argparse
from merger.parallax_estimator.similarity_estimator import compute_distances, fit_minmax_distance

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', '-n', type=str, required=True)
	parser.add_argument('--mass', '-m', type=float, required=True)
	parser.add_argument('--nb_samples', '-nbs', type=int, required=True)

	args = parser.parse_args()

	output_name = args.name
	mass = args.mass
	nb_samples = args.nb_samples

	compute_distances(output_name, distance=fit_minmax_distance, mass=mass, nb_samples=nb_samples)