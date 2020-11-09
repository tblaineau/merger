import argparse
import logging
import pandas as pd
import numpy as np
import time
import os
from merger.clean.libraries import iminuit_fitter
from merger.clean.libraries.merger_library import COLOR_FILTERS
from merger.clean.libraries.parameter_generator import MicrolensingGenerator, microlens_parallax, microlens_simple
from merger.clean.libraries.differential_evolution import fit_ml_de_simple
import matplotlib.pyplot as plt

import scipy.interpolate


def clean_lightcurves(df):
	"""Clean dataframe from bad errors."""

	# magnitude should have already been cleaned but we never know
	df = df.replace(to_replace=[99.999, -99.], value=np.nan).dropna(axis=0, how='all', subset=list(COLOR_FILTERS.keys()))
	for c in COLOR_FILTERS.values():
		df.loc[~df[c['err']].between(0, 9.999, inclusive=False), [c['mag'], c['err']]] = np.nan
	#drop line without information
	df.dropna(how='all', subset=list(COLOR_FILTERS.keys()), inplace=True)
	return df


def mulens_simple(t, mag, u0, t0, tE):
	u = np.sqrt(u0 * u0 + ((t - t0) ** 2) / tE / tE)
	amp = (u ** 2 + 2) / (u * np.sqrt(u ** 2 + 4))
	return - 2.5 * np.log10(amp) + mag


class UniformGenerator:
	def __init__(self, u0_range=[0, 1], tE_range=[10, 500], blend_range=None, seed=1234):
		self.u0_range = np.array(u0_range)
		self.tE_range = np.log10(tE_range)
		self.rdm = np.random.RandomState(seed)
		if blend_range:
			self.blend_range = np.array(blend_range)

	def generate_parameters(self, t0_range):
		if len(np.array(t0_range).shape) == 1:
			size = None
		else:
			size = len(t0_range[0])
		params = {}
		params["u0"] = self.rdm.uniform(*self.u0_range, size=size)
		params["tE"] = np.power(10, self.rdm.uniform(*self.tE_range, size=size))
		params["t0"] = self.rdm.uniform(t0_range[0], t0_range[1])
		if hasattr(self, "blend_range"):
			b = self.rdm.uniform(*self.blend_range, size=size)
			for key in COLOR_FILTERS.keys():
				params["blend_"+key] = b
		return params


class ErrorMagnitudeRelation:
	def __init__(self, df, filters, bin_number=10):
		self.bin_edges = {}
		self.bin_values = {}
		self.filters = filters
		for c in self.filters:
			mask = df[COLOR_FILTERS[c]["mag"]].notnull() & df[COLOR_FILTERS[c]["err"]].notnull() & df[
				COLOR_FILTERS[c]["err"]].between(0, 9.999, inclusive=False)
			df[f"bins_{c}"], self.bin_edges[c] = pd.qcut(df[COLOR_FILTERS[c]["mag"]][mask], bin_number, retbins=True)
			self.bin_values[c] = df[mask].groupby(f"bins_{c}")[COLOR_FILTERS[c]["err"]].agg(np.median).values
			df.drop(columns=f"bins_{c}", inplace=True)

	def get_sigma(self, cfilter, mag):
		"""Get median sigma"""
		index = np.digitize(mag, self.bin_edges[cfilter])
		index = np.where(index >= len(self.bin_edges[cfilter]) - 2, len(self.bin_values[cfilter]) - 1, index - 1)
		return self.bin_values[cfilter][index]

	def vectorized_get_sigma(self, filter_list, mag):
		conds = [filter_list == cfilter for cfilter in self.filters]
		choices = []
		for cfilter in self.filters:
			index = np.digitize(mag, self.bin_edges[cfilter])
			index = np.where(index >= len(self.bin_edges[cfilter]) - 2, len(self.bin_values[cfilter]) - 1, index - 1)
			choices.append(self.bin_values[cfilter][index])
		return np.select(conds, choices)

	def display(self):
		for c in self.filters:
			eds = self.bin_edges[c]
			vals = self.bin_values[c]
			plt.scatter((eds[:-1] + eds[1:]) / 2, vals, s=10)
			print(vals)
			plt.ylim(0, np.nanmax(vals))
			plt.title(c)
			plt.xlabel("Magnitude")
			plt.ylabel("Error")
			plt.show()


def generate_microlensing_events(subdf, sigmag, generator, parallax=False):
	""" For a given star, generate µ-lens event

	[description]

	Arguments:
		subdf {dataframe} -- Dataframe containing only one star
		sigmag {Sigma_Baseline} -- object containg informations on distribution of standard deviation in function of magnitude, for all filters
		raw_stats_df {dataframe} -- Dataframe containing all means and std deviation of the catalogue

	Keyword Arguments:
		blending {bool} -- Simulate blending or not (default: {False})
		parallax {bool} -- Simulate parallax or not (defualt: {False})

	Returns:
		DataFrame -- New star with a µlens event (contains two new columns per filter : 1 for magnitudes, 1 for errors)
	"""
	current_id = subdf["id_E"].iloc[0]

	# Generate µlens parameters
	params = generator.generate_parameters(current_id)
	u0, t0, tE, blend_factors, delta_u, theta = params["u0"], params["t0"], params["tE"], params["blend"], params[
		"delta_u"], params["theta"]

	time = subdf.time.values

	for key, color_filter in COLOR_FILTERS.items():
		mag_i = subdf[color_filter['mag']].values
		mag_med = np.nanmedian(mag_i)
		if parallax:
			mag_th = microlens_parallax(time, mag_med, blend_factors[key], u0, t0, tE, delta_u, theta)
		else:
			mag_th = microlens_simple(time, mag_med, blend_factors[key], u0, t0, tE, delta_u, theta)
		# mag_th is the lightcurve with perfect measurements (computed from original baseline)

		norm = sigmag.get_sigma(key, mag_med) / sigmag.get_sigma(key, mag_th)

		subdf['ampli_' + color_filter['err']] = subdf[color_filter['err']].values / norm
		subdf['ampli_' + color_filter['mag']] = mag_th + (mag_i - mag_med) / norm

	return subdf

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path_to_merged', '-p', type=str, required=True, help="Path to the compressed merged file (ex : 34_78.bz2)")
	parser.add_argument('--output_path', '-o', type=str, default=".", required=False)
	parser.add_argument('--verbose', '-v', action='store_true', help='Debug logging level')
	parser.add_argument('--bad_times_directory', '-btd', type=str, required=False)
	parser.add_argument('--seed', type=int, required=False, default=1234)

	args = parser.parse_args()
	path_to_merged = args.path_to_merged
	output_path = args.output_path
	verbose = args.verbose
	MACHO_bad_times_directory = args.bad_times_directory
	seed = args.seed

	fraction = 0.05
	np.random.seed(seed)

	if verbose:
		logging.basicConfig(level=logging.INFO)

	MACHO_field = path_to_merged.split("/")[-1].split("_")[0]
	t = path_to_merged.split("/")[-1].split("_")[1].split(".")[0]

	merged = pd.read_pickle(path_to_merged, compression='bz2')
	print(len(merged))
	#merged = merged.iloc[:100000]

	logging.info("Lightcurves loaded.")

	st1 = time.time()
	merged = merged.groupby(["id_E", "id_M"]).filter(lambda x: np.random.random()<fraction)
	logging.info(f"Selecting {fraction*100.} % ({len(merged.id_E.unique())}) of lcs in {time.time()-st1:0.3f} seconds")

	merged = clean_lightcurves(merged).reset_index(drop=True)
	sh = ErrorMagnitudeRelation(merged, list(COLOR_FILTERS.keys()), bin_number=20)

	t0_ranges = merged.groupby(["id_E", "id_M"])["time"].agg(["max", "min"]).values.T
	merged = merged.sort_values(["id_E", "id_M"])
	#mg = MicrolensingGenerator(xvt_file=1000000, seed=1234, trange=t0_ranges, u_max=2, max_blend=1., min_blend=0.)
	mg = UniformGenerator(u0_range=[0, 2], tE_range=[1, 3000], blend_range=[0, 1])
	params = mg.generate_parameters(t0_range=t0_ranges) #mass=30)
	cnt = merged.groupby(["id_E", "id_M"])["time"].agg(len).values

	#Save true_parameters
	true_parameters = pd.concat([pd.DataFrame(params), merged[["id_E", "id_M"]].drop_duplicates(ignore_index=True)], axis=1)
	true_parameters.to_pickle(os.path.join(output_path, "truth_"+str(MACHO_field) + "_" + str(t) + ".pkl"))

	for key in params.keys():
		merged[key] = np.repeat(params[key], cnt)
		mag_th = dict()
		norm = dict()

	for key in COLOR_FILTERS.keys():
		merged["mag_median_" + key] = merged[["id_E", "id_M", COLOR_FILTERS[key]["mag"]]].groupby(["id_E", "id_M"]).transform("median")
		#mag_th[key] = microlens_parallax(merged.time.values, merged["mag_median_" + key].values, merged["blend_"+key].values, merged.u0.values,
		#								 merged.t0.values, merged.tE.values, merged.delta_u.values, merged.theta.values)
		mag_th[key] = microlens_simple(merged.time.values, merged["mag_median_" + key].values, merged["blend_"+key].values, merged.u0.values, merged.t0.values, merged.tE.values, 0, 0)
		norm[key] = sh.vectorized_get_sigma(key, merged["mag_median_" + key].values) / sh.vectorized_get_sigma(key, mag_th[key])
		merged["new_" + COLOR_FILTERS[key]["err"]] = merged[COLOR_FILTERS[key]["err"]] / norm[key]
		merged["new_" + COLOR_FILTERS[key]["mag"]] = mag_th[key] + (merged[COLOR_FILTERS[key]["mag"]] - merged["mag_median_" + key]) / norm[key]

	for key in COLOR_FILTERS.keys():
		merged = merged.rename(columns={COLOR_FILTERS[key]["mag"]: "old_" + COLOR_FILTERS[key]["mag"],
										COLOR_FILTERS[key]["err"]: "old_" + COLOR_FILTERS[key]["err"],
										"new_" + COLOR_FILTERS[key]["mag"]: COLOR_FILTERS[key]["mag"],
										"new_" + COLOR_FILTERS[key]["err"]: COLOR_FILTERS[key]["err"]})

	logging.info("Bad time removal.")
	dfr = []
	dfb = []
	#MACHO_bad_times_directory = "/Users/tristanblaineau/tmp2"
	MACHO_btt = 0.1

	try:
		df = pd.DataFrame(np.load(os.path.join(MACHO_bad_times_directory, str(MACHO_field) + "_red_M_ratios.npy")),
						  columns=["red_amp", "time", "ratio"])
		df.loc[:, "field"] = MACHO_field
		dfr.append(df)
		df = pd.DataFrame(np.load(os.path.join(MACHO_bad_times_directory, str(MACHO_field) + "_blue_M_ratios.npy")),
						  columns=["blue_amp", "time", "ratio"])
		df.loc[:, "field"] = MACHO_field
		dfb.append(df)
	except FileNotFoundError:
		logging.warning("No ratio file found for field " + str(MACHO_field) + ".")
	else:
		merged.reset_index(drop=True, inplace=True)

		dfr = pd.concat(dfr)
		dfb = pd.concat(dfb)

		pms = list(zip(merged["time"].values, merged["red_amp"].values))
		pdf = list(zip(dfr[dfr.ratio > MACHO_btt]["time"].values, dfr[dfr.ratio > MACHO_btt]["red_amp"].values))
		result = pd.Series(pms).isin(pdf)
		merged[result].red_M = np.nan
		merged[result].rederr_M = np.nan

		pms = list(zip(merged["time"].values, merged["blue_amp"].values))
		pdf = list(zip(dfb[dfb.ratio > MACHO_btt]["time"].values, dfb[dfb.ratio > MACHO_btt]["blue_amp"].values))
		result = pd.Series(pms).isin(pdf)
		merged[result].blue_M = np.nan
		merged[result].blueerr_M = np.nan

		merged = merged.dropna(axis=0, how='all', subset=['blue_E', 'red_E', 'blue_M', 'red_M'])

	# ndf = merged.merge(merged.groupby(["id_E", "id_M"]).median(), on=["id_E", "id_M"], suffixes=("", "_median"))

	#ndf = ndf.dropna(subset=["red_E", "rederr_E"])
	# for key in COLOR_FILTERS.keys():
	# 	_, bins = pd.qcut(ndf[COLOR_FILTERS[key]["mag"]], 30, retbins=True)
	# 	ndf.loc[:, "dist"] = (ndf[COLOR_FILTERS[key]["mag"]] - ndf[COLOR_FILTERS[key]["mag"]+"_median"]) / ndf[COLOR_FILTERS[key]["err"]]
	# 	fs = []
	# 	for i in zip(bins[:-1], bins[1:]):
	# 		t = ndf[ndf.red_E.between(*i)]
	# 		fs.append(np.std(t.dist[np.abs(t.dist) < 10].values - t.dist.median()))
	# 	interpolation = scipy.interpolate.interp1d((bins[:-1] + bins[1:])[:] / 2, fs[:], fill_value=(fs[0], fs[-1]), bounds_error=False)
	# 	df.rederr_E = df.rederr_E * interpolation(df.red_E)
	#ndf = ndf[(ndf.red_E < 23) & (ndf.red_E > 14)]

	logging.info("Done.")
	logging.info("Starting fit")
	iminuit_fitter.fit_all(merged=merged,
						   filename=str(MACHO_field) + "_" + str(t) + ".pkl",
						   input_dir_path=output_path,
						   output_dir_path=output_path,
						   fit_function=fit_ml_de_simple,
						   do_cut5=True,
						   hesse=True)
