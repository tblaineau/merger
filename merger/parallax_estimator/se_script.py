
import argparse
from merger.parallax_estimator.similarity_estimator import compute_distances, minmax_distance_scipy

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', '-n', type=str, required=True)
	parser.add_argument('--mass', '-m', type=float, required=False)
	parser.add_argument('--nb_samples_job', '-nsj', type=int, required=True)
	parser.add_argument('--current_job', '-cj', type=int, required=True)

	args = parser.parse_args()

	output_name = args.name
	mass = args.mass


	nb_samples_job = args.nb_samples_job
	current_job = args.current_job

	start = (current_job-1) * nb_samples_job
	end = current_job * nb_samples_job

	if mass is None:
		mass = [0.1, 1, 10, 30, 100]
		for cmass in mass:
			compute_distances(output_name, distance=minmax_distance_scipy, mass=mass, start=start, end=end)