
import argparse
import os
import numpy as np
from merger.parallax_estimator.similarity_estimator import compute_distances, minmax_distance_scipy2

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', '-n', type=str, required=True)
	parser.add_argument('--parameter_file', '-pf', type=str, required=True)
	parser.add_argument('--nb_samples_job', '-nsj', type=int, required=True)
	parser.add_argument('--current_job', '-cj', type=int, required=True)

	args = parser.parse_args()

	output_name = args.name
	parameter_file = args.parameter_file
	nb_samples_job = args.nb_samples_job
	current_job = args.current_job

	assert os.path.isfile(parameter_file), 'Parameter file does not exist.'
	assert nb_samples_job>0, 'Invalid number of samples per job.'
	assert current_job>0, 'Invalid current job number.'

	start = (current_job-1) * nb_samples_job
	end = current_job * nb_samples_job

	compute_distances(output_name, distance=minmax_distance_scipy2, parameter_list=np.load(parameter_file, allow_pickle=True), start=start, end=end)
