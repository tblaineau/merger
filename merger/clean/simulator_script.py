import argparse
import logging
import pandas as pd
import numpy as np
import time
import os
from merger.clean.libraries import differential_evolution as de
from merger.clean.libraries.merger_library import COLOR_FILTERS
from merger.clean.libraries.parameter_generator import microlens_parallax, microlens_simple, delta_u_from_x, tE_from_xvt
import matplotlib.pyplot as plt
import numba as nb
import scipy.interpolate


@nb.njit
def find_nearest(array, values):
	o = []
	for i in range(len(values)):
		o.append((np.abs(array - values[i])).argmin())
	return o


def clean_lightcurves(df):
	"""Clean dataframe from bad errors."""

	# magnitude should have already been cleaned but we never know
	df = df.replace(to_replace=[99.999, -99.], value=np.nan).dropna(axis=0, how='all', subset=list(COLOR_FILTERS.keys()))
	for c in COLOR_FILTERS.values():
		df.loc[~df[c['err']].between(0, 9.999, inclusive=False), [c['mag'], c['err']]] = np.nan
	#drop line without information
	df.dropna(how='all', subset=list(COLOR_FILTERS.keys()), inplace=True)
	return df


@nb.njit
def amp(t, b, u0, t0, tE):
	u = np.sqrt(u0 * u0 + ((t - t0) ** 2) / tE / tE)
	return (u ** 2 + 2) / (u * np.sqrt(u ** 2 + 4)) * (1-b) + b


class RealisticGenerator:
	"""
	Class to generate microlensing paramters

	Parameters
	----------
	id_E_list : list
		List of EROS index to compute density
	xvt_file : str or int
		If a str : Path to file containing x - v_T pairs generated through the Hasting-Metropolis algorithm
		If int, number of xvt pairs to pool
	seed : int
		Seed used for numpy.seed
	tmin : float
		lower limit of t_0
	tmax : float
		upper limits of t_0
	max_blend : float
		maximum authorized blend, if max_blend=0, no blending
	"""

	def __init__(self, medlist, xvt_file=None, seed=1234, tmin=48928., tmax=52697., u_max=2., max_blend=0.001,
				 blend_directory=None,
				 densities_path="/pbs/home/b/blaineau/work/merger/merger/clean/useful_files/"):
		self.seed = seed
		self.xvt_file = xvt_file
		self.rdm = np.random.RandomState(seed)
		self.tmin = tmin
		self.tmax = tmax
		self.u_max = u_max
		self.max_blend = max_blend
		self.blending = bool(blend_directory)
		self.blend_pdf = None
		self.generate_mass = False

		if not self.blending:
			self.w = [1] * len(medlist)
			self.blends = {
				"frac_red_E": [1] * len(medlist),
				"frac_blue_E": [1] * len(medlist),
				"frac_red_M": [1] * len(medlist),
				"frac_blue_M": [1] * len(medlist)
			}
		elif not os.path.exists(blend_directory):
			logging.error("Invalid blend factors directory")
		else:
			# load fracs catalogs
			fracs_catalogues = [pd.read_csv(os.path.join(blend_directory, "sparse_42.csv")),
									 pd.read_csv(os.path.join(blend_directory, "medium_67.csv"))]
			fracs_catalogues = [fcat[fcat.frac_red_E.values > self.max_blend].reset_index(drop=True) for fcat in fracs_catalogues]
			mags_catalogues = [fcat.groupby("index_eros").blue_E.agg([max, "size"]) for fcat in fracs_catalogues]

			# densité des champs EROS
			try:
				densities_E = pd.DataFrame(np.loadtxt(os.path.join(densities_path, "densities_E.txt")),
										   columns=["field", "ccd", "density"])
			except OSError:
				logging.error("Density file not found :", os.path.join(densities_path, "densities_E.txt"))
			id_E_list = medlist.index.get_level_values("id_E")
			densities_E = densities_E.append(dict(field=99, ccd=99, density=np.nan),
											 ignore_index=True)  # account for missing idE
			fields = id_E_list.str[2:5].values
			fields = np.where(fields == "_id", 99, fields).astype(int)  # account for missing idE
			ccds = id_E_list.str[5].values
			ccds = np.where(ccds == "_", 99, ccds).astype(int)  # account for missing idE
			densities_E = densities_E.set_index(["field", "ccd"]).loc[list(zip(fields, ccds))].values

			# densité des champs MACHO
			try:
				densities_M = pd.DataFrame(np.loadtxt(os.path.join(densities_path, "densities_M.txt")),
										   columns=["density", "field"])
			except OSError:
				logging.error("Density file not found :", os.path.join(densities_path, "densities_M.txt"))
			densities_M.loc[:, "field"] = densities_M["field"].astype(int)
			densities_M = densities_M.append(dict(density=np.nan, field="_id"), ignore_index=True)
			densities_M.loc[:, "field"] = densities_M["field"].astype(str)
			id_M_list = medlist.reset_index()["id_M"]
			fields = id_M_list.str.split(":", expand=True)[0].values.astype(str)
			densities_M = densities_M.set_index(["field"]).loc[fields].values[:, 0]

			self.blends = []
			self.w = []
			for i, (index, row) in enumerate(medlist.iterrows()):
				if index[0] != "no_id_E":
					if densities_E[i] > 67:
						index_catalogue = 1
					else:
						index_catalogue = 0
				elif index[1] != "no_id_M":
					if densities_M[i] > 92:
						index_catalogue = 1
					else:
						index_catalogue = 0
				else:
					logging.error("Both eros and macho have no index ?")

				# ccolor est la couleur considérée quand on recherche l'étoile la plus proche
				for key in COLOR_FILTERS.keys():
					if not np.isnan(row[key]):
						ccolor = key
						break

				# index de l'étoile avec la magnitude la plus proche
				idx_in_cat = find_nearest(mags_catalogues[index_catalogue][ccolor, "max"].values, [row[ccolor]])
				# numéro eros correspondant à cette étoile
				eidx = mags_catalogues[index_catalogue].index[idx_in_cat]
				#
				imags = mags_catalogues[index_catalogue]
				# nombre d'étoiles HST pour l'étoile E/M
				w = imags.loc[imags.index[idx_in_cat]][ccolor, "size"].values[0]
				#
				fcat = fracs_catalogues[index_catalogue]
				# dictionnaire numéro eros -> index iloc dans le fcat
				iero_to_loc = {v: k for k, v in dict(fcat.drop_duplicates("index_eros").index_eros).items()}
				# index iloc de l'étoile E/M
				eloc = iero_to_loc[eidx[0]]
				# facteurs de blend à partir de l'index iloc pour toutes les étoiles HST
				for j in range(w):
					self.blends.append(fcat.iloc[eloc + j][["frac_red_E", "frac_blue_E", "frac_red_M", "frac_blue_M"]])
				# print(catfb)
				self.w.append(w)
		self.blends = pd.DataFrame(self.blends)
		self.w = np.array(self.w)
		self.weights = 1 / self.w

		if self.xvt_file:
			if isinstance(self.xvt_file, str):
				try:
					self.xvts = np.load(self.xvt_file)
				except FileNotFoundError:
					logging.error(f"xvt file not found : {self.xvt_file}")
			else:
				logging.error(f"xvts can't be loaded or generated, check variable : {self.xvt_file}")

	def set_seed(self, new_seed):
		self.seed = new_seed
		self.rdm = np.random.RandomState(new_seed)

	def generate_parameters(self, mass=30., nb_parameters=1, t0_ranges=None):
		"""
		Generate a set of microlensing parameters, including parallax and blending using S-model and fixed mass

		Parameters
		----------
		seed : str
			Seed used for parameter generation (EROS id)
		mass : float
			mass for which generate paramters (\implies \delta_u, t_E)
		nb_parameters : int
			number of parameters set to generate, set to t0_ranges.shape[1] if using t0_ranges
		t0_ranges : array (2, n)
			[tmin, tmax] array. t0 will be generated between tmin-2tE and tmax+2tE
		Returns
		-------
		dict
			Dictionnary of lists containing the parameters set
		"""

		if self.generate_mass:
			mass = self.rdm.uniform(1, 1000, size=nb_parameters)
		else:
			mass = np.array([mass] * nb_parameters)
		u0 = self.rdm.uniform(0, self.u_max, size=nb_parameters)
		x, vt, theta = (self.xvts.T[self.rdm.randint(0, self.xvts.shape[1], size=nb_parameters)]).T
		vt *= self.rdm.choice([-1., 1.], size=nb_parameters, replace=True)
		delta_u = delta_u_from_x(x, mass=mass)
		tE = tE_from_xvt(x, vt, mass=mass)
		if not t0_ranges is None:
			t0 = self.rdm.uniform(np.array(t0_ranges[0]) - 2 * abs(tE), np.array(t0_ranges[1]) + 2 * abs(tE),
								  size=nb_parameters)
		else:
			t0 = self.rdm.uniform(self.tmin - 2 * abs(tE), self.tmax + 2 * abs(tE), size=nb_parameters)
		params = {
			'u0': np.repeat(u0, self.w),
			't0': np.repeat(t0, self.w),
			'tE': np.repeat(tE, self.w),
			'delta_u': np.repeat(delta_u, self.w),
			'theta': np.repeat(theta, self.w),
			'mass': np.repeat(mass, self.w),
			'x': np.repeat(x, self.w),
			'vt': np.repeat(vt, self.w),
			'tmin': np.repeat(np.array(t0_ranges[0]), self.w),
			'tmax': np.repeat(np.array(t0_ranges[1]), self.w),
		}

		for key in COLOR_FILTERS.keys():
			if self.blending:
				params['blend_' + key] = self.blends["frac_" + key].values
				params['weight'] = self.weights
			else:
				params['blend_' + key] = [1] * nb_parameters
				params["weight"] = [1] * nb_parameters
		return params, self.w


class UniformGenerator:
	def __init__(self, u0_range=[0, 1], tE_range=[10, 500], blend_range=None, seed=1234):
		self.u0_range = np.array(u0_range)
		self.tE_range = np.log10(tE_range)
		self.rdm = np.random.RandomState(seed)
		if blend_range:
			self.blend_range = np.array(blend_range)

	def generate_parameters(self, t0_ranges):
		if len(np.array(t0_ranges).shape) == 1:
			size = None
		else:
			size = len(t0_ranges[0])
		params = {}
		params["u0"] = self.rdm.uniform(*self.u0_range, size=size)
		params["tE"] = np.power(10, self.rdm.uniform(*self.tE_range, size=size))
		params["t0"] = self.rdm.uniform(t0_ranges[0], t0_ranges[1])
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
	parser.add_argument('--path_to_merged', '-p', type=str, required=True, help="Path to the compressed merged file (ex : 1415000_1420000.parquet)")
	parser.add_argument('--output_path', '-o', type=str, default=".", required=False)
	parser.add_argument('--verbose', '-v', action='store_true', help='Debug logging level')
	parser.add_argument('--seed', type=int, required=False, default=1234)
	parser.add_argument('--fraction', '-f', type=float, required=False, default=0.05, help="Fraction of lightcurves used for simulation")
	parser.add_argument('--mass', '-m', type=float, required=False, default=100, help="Mass of lenses")

	args = parser.parse_args()
	path_to_merged = args.path_to_merged
	output_path = args.output_path
	verbose = args.verbose
	seed = args.seed
	fraction = args.fraction
	mass = args.mass

	current_filename = path_to_merged.split("/")[-1].split(".")[0]

	np.random.seed(seed)

	if verbose:
		logging.basicConfig(level=logging.INFO)

	merged = pd.read_parquet(path_to_merged)

	logging.info("Lightcurves loaded.")

	merged["id_M"].replace(to_replace=[None], value="no_id_M", inplace=True)
	merged["id_E"].replace(to_replace=[None], value="no_id_E", inplace=True)

	# ufilters = used filters (filter with at least 1 non nan measurement)
	ufilters = []
	for key in COLOR_FILTERS.keys():
		if not (~merged[key].isnull()).sum() < 10:
			ufilters.append(key)

	# construction des erreurs sur l'ensemble des données pour avoir plus de stats
	# à faire par champs et charger ?
	sh = ErrorMagnitudeRelation(merged, ufilters, bin_number=20)

	# selection aléatoire des courbes
	st1 = time.time()
	merged = merged.groupby(["id_E", "id_M"]).filter(lambda x: np.random.random() < fraction)
	logging.info(
		f"Selecting {fraction * 100.} % ({len(merged.id_E.unique())}) of lcs in {time.time() - st1:0.3f} seconds")

	# nettoyage des courbes de lumière
	merged = clean_lightcurves(merged).reset_index(drop=True)

	t0_ranges = merged.groupby(["id_E", "id_M"])["time"].agg(["min", "max"]).values.T
	merged = merged.sort_values(["id_E", "id_M"])

	# magntiudes moyennes/médianes/de base des étoiles
	baselines = merged.groupby(["id_E", "id_M"])[list(COLOR_FILTERS.keys())].agg(np.median)
	# quand il n'y a aucune mesure la colonne n'eiste pas : nous la créons
	for key in list(COLOR_FILTERS.keys()):
		if key not in baselines.columns:
			baselines.loc[:, key] = np.nan

	# fix column order
	baselines = baselines[["blue_E", "red_E", "blue_M", "red_M"]]
	baselines = baselines.reorder_levels(["id_E", "id_M"])
	mg = RealisticGenerator(baselines, u_max=1.5, seed=seed, max_blend=0.01,
							blend_directory="/pbs/home/b/blaineau/work/simulation_prod/useful_files",
							xvt_file="/pbs/home/b/blaineau/work/simulation_prod/useful_files/xvt_clean.npy",
							densities_path="/pbs/home/b/blaineau/work/simulation_prod/useful_files"
							)

	params, w = mg.generate_parameters(t0_ranges=t0_ranges, nb_parameters=t0_ranges.shape[1], mass=mass)
	cnt = merged.groupby(["id_E", "id_M"])["time"].agg(len).values.astype(int)
	cm = np.cumsum(cnt)-cnt
	merged = merged.iloc[np.concatenate(np.repeat([cm[i]+np.arange(cnt[i]) for i in range(len(cnt))], w))]
	merged.loc[:,"id_E"] = merged.id_E.str.cat(np.repeat(np.arange(len(params["u0"])), np.repeat(cnt, w)).astype(str), sep="_")
	#Save true_parameters
	true_parameters = pd.concat([pd.DataFrame(params), merged[["id_E", "id_M"]].drop_duplicates(ignore_index=True)], axis=1)
	
	for key in params.keys():
		merged[key] = np.repeat(params[key], np.repeat(cnt, w))
		mag_th = dict()
		norm = dict()

	for key in COLOR_FILTERS.keys():
		merged["mag_median_" + key] = merged[["id_E", "id_M", COLOR_FILTERS[key]["mag"]]].groupby(["id_E", "id_M"]).transform("median")
		#mag_th[key] = microlens_parallax(merged.time.values, merged["mag_median_" + key].values, merged["blend_"+key].values, merged.u0.values,
		#								 merged.t0.values, merged.tE.values, merged.delta_u.values, merged.theta.values)
		mag_th[key] = microlens_parallax(merged.time.values, merged["mag_median_" + key].values, 1-merged["blend_"+key].values, merged.u0.values, merged.t0.values, merged.tE.values, merged.delta_u.values, merged.theta.values)
		norm[key] = sh.vectorized_get_sigma(key, merged["mag_median_" + key].values) / sh.vectorized_get_sigma(key, mag_th[key])
		merged["new_" + COLOR_FILTERS[key]["err"]] = merged[COLOR_FILTERS[key]["err"]] / norm[key]
		merged["new_" + COLOR_FILTERS[key]["mag"]] = mag_th[key] + (merged[COLOR_FILTERS[key]["mag"]] - merged["mag_median_" + key]) / norm[key]

	for key in COLOR_FILTERS.keys():
		merged = merged.rename(columns={COLOR_FILTERS[key]["mag"]: "old_" + COLOR_FILTERS[key]["mag"],
										COLOR_FILTERS[key]["err"]: "old_" + COLOR_FILTERS[key]["err"],
										"new_" + COLOR_FILTERS[key]["mag"]: COLOR_FILTERS[key]["mag"],
										"new_" + COLOR_FILTERS[key]["err"]: COLOR_FILTERS[key]["err"]})

	merged.loc[:, "amp"] = amp(merged.time.values, merged["blend_"+key].values, merged.u0.values, merged.t0.values, merged.tE.values)
	merged.loc[:, "sup2"] = merged.amp > amp(0, 0, 2, 0, 1)
	merged.loc[:, "sup1"] = merged.amp > amp(0, 0, 1, 0, 1)
	true_parameters = true_parameters.merge(merged.groupby(["id_E", "id_M"])["sup1"].agg(sum), left_on=["id_E", "id_M"], right_index=True)
	true_parameters = true_parameters.merge(merged.groupby(["id_E", "id_M"])["sup2"].agg(sum), left_on=["id_E", "id_M"], right_index=True)
	true_parameters.to_parquet(os.path.join(output_path, "truth_" + str(current_filename) + ".parquet"), index=False)

	logging.info("Done.")
	logging.info("Starting fit")
	res = merged.groupby(["id_M", "id_E"]).apply(de.fit_ml_de_flux, do_cut5=True, minos=True)
	res.to_csv(os.path.join(output_path, "res_" + str(current_filename) + ".csv"))
