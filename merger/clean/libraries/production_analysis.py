
import os
import pandas as pd
from IPython.display import clear_output
from merger.clean.libraries.merger_library import COLOR_FILTERS

def update_progress(progress):
	bar_length = 40
	if isinstance(progress, int):
		progress = float(progress)
	if not isinstance(progress, float):
		progress = 0
	if progress < 0:
		progress = 0
	if progress >= 1:
		progress = 1

	block = int(round(bar_length * progress))

	clear_output(wait = True)
	text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
	print(text)

def load_results(dirpath):
	pds = []
	for (dirpath, dirnames, filenames) in os.walk(dirpath):
		for i, filename in enumerate(filenames):
			if filename[-4:]=='.pkl':
				pds.append(pd.read_pickle(os.path.join(dirpath,filename)))
			update_progress(i/len(filenames))
	return pd.concat(pds)

def compute_values(all_params, color_filters=COLOR_FILTERS):
	all_params["dof"] = sum([all_params["counts_" + key] for key in color_filters.keys()])
	for afilter in color_filters.keys():
		all_params['reduced_micro_chi2_' + afilter] = all_params['micro_chi2_' + afilter] / (
					all_params['counts_' + afilter] - 1.75)
		all_params['reduced_flat_chi2_' + afilter] = all_params['flat_chi2_' + afilter] / (
					all_params['counts_' + afilter] - 1.75)
		all_params["delta_chi2_" + afilter] = (all_params['flat_chi2_' + afilter] - all_params[
			'micro_chi2_' + afilter]) / all_params['micro_chi2_' + afilter]
		all_params['full_delta_chi2_' + afilter] = all_params['delta_chi2_' + afilter] * np.sqrt(
			(all_params['counts_' + afilter] - 1.75) / 2.)
	all_params["delta_chi2"] = (all_params.flat_fval - all_params.micro_fval) / all_params.micro_fval
	all_params['full_delta_chi2'] = all_params.delta_chi2 * np.sqrt((all_params.dof - 7) / 2.)
	all_params['reduced_micro_chi2'] = all_params.micro_fval / (all_params.dof - 7)
	all_params['reduced_flat_chi2'] = all_params.flat_fval / (all_params.dof - 7)
	return all_params