import pandas as pd
import numpy as np
import os, ssl, logging, glob
from irods.session import iRODSSession
from irods.exception import CollectionDoesNotExist, DataObjectDoesNotExist
import tarfile
from io import StringIO
import time


def load_irods_eros_lightcurves(irods_filepath="", idE_list=[]):
	#irods_filepath = '/eros/data/eros2/lightcurves/lm/lm041/lm0412/lm0412k-lc.tar.gz'
	try:
		env_file = os.environ['IRODS_ENVIRONMENT_FILE']
	except KeyError:
		env_file = os.path.expanduser('~/.irods/irods_environment.json')

	ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH, cafile=None, capath=None, cadata=None)
	ssl_settings = {'ssl_context': ssl_context}
	pds = []
	with iRODSSession(irods_env_file=env_file, **ssl_settings) as session:
		try:
			coll = session.data_objects.get(irods_filepath)
		except CollectionDoesNotExist:
			logging.error(f"iRods path not found : {irods_filepath}")
		else:
			g = []
			with tarfile.open(fileobj=coll.open("r"), mode="r:gz") as f:
				f.next()
				data = f.next()
				while data is not None:
					if len(idE_list)>0 and data.name in idE_list:
						g.append([data.name, f.extractfile(data).read()])
					data = f.next()

			for name, l in g:
				df = pd.read_csv(StringIO(str(l, 'utf-8')), delim_whitespace=True, comment="#",
								 names=["hjd", "mag_R", "magerr_R", "mag_B", "magerr_B"])
				df.loc[:, "id_E"] = name[8:-5]
				pds.append(df)

			pds = pd.concat(pds)
	return pds

def h(xi, xj, xm, p):
	if xi==xj:
		return np.sign(p-i-j-1)
	else:
		return ((xi - xm) - (xm - xj))/(xi - xj)


def merger_small_sample(output_dir_path, start, end,
					  correspondance_file_path = "/pbs/home/b/blaineau/work/notebooks/combined.parquet",
					  macho_files_path="/sps/eros/users/blaineau/small_sample/macho/sample_macho.parquet",
					  eros_files_path = "/sps/eros/users/blaineau/small_sample/eros/sample_eros.parquet",
					  eros_ratio_path = "/pbs/home/b/blaineau/work/notebooks/eros_cleaning/ratios",
					  macho_ratio_path = "/pbs/home/b/blaineau/work/bad_times/bt_macho",
					  save=True,
					  verbose=False):
	if verbose:
		logging.basicConfig(level=logging.INFO)

	st1 = time.time()
	# Load combined ids
	logging.info("Reading correspondance file.")
	ids = pd.read_parquet(correspondance_file_path, columns=["id_M", "id_E"])
	sample_ids = np.loadtxt("/pbs/home/b/blaineau/work/simulation_prod/sample.csv", dtype=str)
	sample_ids = pd.DataFrame(sample_ids, columns=["id_E", "id_M"])
	ids = ids[(ids.id_E.isin(sample_ids.id_E) | ids.id_M.isin(sample_ids.id_M))]

	id_Es = ids.id_E

	logging.info("Loading EROS light curves")
	quarts = id_Es.str[:7].unique()
	keep_E = pd.read_parquet(eros_files_path)
	keep_E.time += 49999.5

	# Clean EROS
	logging.info("Cleaning EROS light curves")
	ccds = keep_E.id_E.str[:6]
	# r=0, b=1
	blue_eros_max_ratio = 0.015
	red_eros_max_ratio = 0.017
	for ccd in ccds.unique():
		if ccd is None:
			continue
		ratios = pd.read_parquet(os.path.join(eros_ratio_path, "ratios_" + ccd + ".parquet"))
		for color_ratio, color, color_err in zip(["high_distance_b10", "high_distance_r10"], ["red_E", "blue_E"], ["rederr_E", "blueerr_E"]):
			g = ratios[color_ratio][ratios[color_ratio] > 0]
			x = g.values
			a, xm, b = np.nanquantile(x, q=[0.25, 0.5, 0.75])
			p = len(x[x <= xm])
			hs = []
			for xj in x[x <= xm]:
				for xi in x[x > xm]:
					hs.append(h(xi, xj, xm, p))
			mc = np.median(hs)
			if mc < 0:
				# mc=0
				w3 = b + 1.5 * np.exp(4 * mc) * (b - a)
			else:
				w3 = b + 1.5 * np.exp(3 * mc) * (b - a)

			r = g[(g > w3)].sort_values(ascending=False)
			one_percent = int(np.round(0.01 * len(g)))
			if len(r) > one_percent:
				r = r.iloc[:one_percent]
			keep_E.loc[keep_E["time"].isin(r.index) & (ccds == ccd), [color, color_err]] = np.nan

	# Load and remove JBM times
	logging.info("Removing JBM times")
	discard = pd.read_csv("/pbs/home/b/blaineau/work/notebooks/eros_cleaning/discard.txt", sep="/", usecols=[0, 1, 2, 3, 5], names=["target", "n_ccd", "n_color", "n_quart", "name"])
	discard["n_time"] = discard["name"].str[10:-5]
	times_translation = pd.read_csv("/pbs/home/b/blaineau/work/notebooks/eros_cleaning/datesall/all.csv", usecols=[0, 3], names=["n_time", "hjd"])
	discard = pd.merge(discard, times_translation, on="n_time")
	discard.loc[:, "hjd"] = discard.hjd.astype(float)
	discard = discard[discard["target"]=="lm"].reset_index(drop=True)
	discard.hjd = discard.hjd + 49999.5
	for i, color in enumerate(["red_E", "blue_E"]):
		td = discard[discard.n_color.str[-1]==str(i)]
		keep_E.loc[:, "n_quart"] = keep_E.id_E.str[:5] + str(i) + keep_E.id_E.str[5]
		rm = keep_E[["n_quart", "time"]].isin(discard[["n_quart", "hjd"]]).all(axis=1)
		logging.info("Removed "+str(rm.sum())+" points in "+color)	
		keep_E.loc[rm, color] = np.nan
	keep_E.drop("n_quart", axis=1, inplace=True)

	# droping empty lines
	keep_E.dropna(subset=["red_E", "blue_E"], how="all", inplace=True)
	keep_E = pd.merge(keep_E, ids, on="id_E", how="left")
	
	
	# LOADING MACHO
	logging.info("Loading MACHO light curves.")
	keep_M = pd.read_parquet(macho_files_path).reset_index(drop=True)

	# Cleaning MACHO lcs.
	max_macho_fraction = 0.05
	logging.info("Cleaning MACHO light curves")
	fields = keep_M.id_M.str.split(":").str[0]
	for field in fields.unique():
		dfb = pd.DataFrame(np.load(os.path.join(macho_ratio_path, str(field) + "_blue_M_ratios.npy")), columns=["blue_amp", "time", "ratio"])
		pms = list(zip(keep_M["time"].values, keep_M["blue_amp"].values))
		pdf = list(zip(dfb[dfb.ratio > max_macho_fraction]["time"].values, dfb[dfb.ratio > max_macho_fraction]["blue_amp"].values))
		result = pd.Series(pms).isin(pdf) & (fields == field)
		keep_M.loc[result, "blue_M"] = np.nan
		keep_M.loc[result, "blueerr_M"] = np.nan

		dfr = pd.DataFrame(np.load(os.path.join(macho_ratio_path, str(field) + "_red_M_ratios.npy")), columns=["red_amp", "time", "ratio"])
		pms = list(zip(keep_M["time"].values, keep_M["red_amp"].values))
		pdf = list(zip(dfr[dfr.ratio > max_macho_fraction]["time"].values, dfr[dfr.ratio > max_macho_fraction]["red_amp"].values))
		result = pd.Series(pms).isin(pdf) & (fields == field)
		keep_M.loc[result, "red_M"] = np.nan
		keep_M.loc[result, "rederr_M"] = np.nan

	keep_M.drop(["blue_amp", "red_amp"], axis=1, inplace=True)
	keep_M = pd.merge(keep_M, ids, on="id_M", how="left")

	# MERGING LIGHT CURVES
	logging.info("Merging light curves...")
	merged = pd.concat([keep_E, keep_M])
	del keep_E
	del keep_M

	# save merged dataframe
	if save:
		logging.info("Saving")
		merged.to_parquet(os.path.join(output_dir_path, str(start) + "_" + str(end) + ".parquet"), index=False)

	logging.info("Total merging time : " + str(time.time() - st1))
	return merged


from memory_profiler import profile

@profile
def merger_eros_first(output_dir_path, start, end,
						correspondance_file_path = "/pbs/home/b/blaineau/work/notebooks/combined.parquet",
						macho_files_path="/sps/eros/data/macho/lightcurves/F_",
						eros_files_path = "/sps/eros/users/blaineau/eros_fast_read/",
						eros_ratio_path = "/pbs/home/b/blaineau/work/notebooks/eros_cleaning/ratios",
						macho_ratio_path = "/pbs/home/b/blaineau/work/bad_times/bt_macho",
						jbm_discard_path = "/pbs/home/b/blaineau/work/notebooks/eros_cleaning/discard.txt",
						jbm_date_conversion = "/pbs/home/b/blaineau/work/notebooks/eros_cleaning/datesall/all.csv",
						save=True,
						verbose=False):
	"""
	Merge EROS and MACHO lightcurves, using EROS as starter

	Parameters
	----------
	output_dir_path : str
		Where to put resulting .pkl
	start : int
		Start line in association file
	end : int
		End line in association line
	correspondance_file_path : str
		Path to association files
	eros_files_path : str
		Path to EROS parquet files
	macho_files_path : str
		Path to MACHO files
	eros_ratio_path : str
		Path to EROS ratios files
	macho_ratio_path : str
		Path to MACHO ratios files
	save : bool
		Either to save or not the resulting merged file (saved in outputdir path)
	verbose : bool
		logging level info

	Raises
	------
	NameError
		Did not find common stars in fields

	Returns
	-------
	pd.DataFrame
	"""

	if verbose:
		logging.basicConfig(level=logging.INFO)

	st1 = time.time()
	# Load combined ids
	logging.info("Reading correspondace file.")
	ids = pd.read_parquet(correspondance_file_path, columns=["id_M", "id_E"]).iloc[start:end]
	id_Es = ids.id_E

	# l o a d   E R O S
	logging.info("Loading EROS light curves")
	quarts = id_Es.str[:7].unique()
	keep_E = []
	for quart in quarts:
		logging.info(quart)
		try:
			path = os.path.join(eros_files_path, quart[:5], quart + ".parquet")
		except TypeError:
			continue
		d = pd.read_parquet(path)
		keep_E.append(d[d.id_E.isin(id_Es)])
	del d
	keep_E = pd.concat(keep_E)
	keep_E.time += 49999.5

	#Clean EROS
	logging.info("Cleaning EROS light curves")
	ccds = keep_E.id_E.str[:6]
	# r=0, b=1
	blue_eros_max_ratio = 0.015
	red_eros_max_ratio = 0.017
	for ccd in ccds.unique():
		if ccd is None:
			continue
		ratios = pd.read_parquet(os.path.join(eros_ratio_path, "ratios_" + ccd + ".parquet"))
		for color_ratio, color, color_err in zip(["high_distance_b10", "high_distance_r10"], ["red_E", "blue_E"], ["rederr_E", "blueerr_E"]):
			g = ratios[color_ratio][ratios[color_ratio] > 0]
			x = g.values
			a, xm, b = np.nanquantile(x, q=[0.25, 0.5, 0.75])
			p = len(x[x <= xm])
			hs = []
			for xj in x[x <= xm]:
				for xi in x[x > xm]:
					hs.append(h(xi, xj, xm, p))
			mc = np.median(hs)
			if mc < 0:
				# mc=0
				w3 = b + 1.5 * np.exp(4 * mc) * (b - a)
			else:
				w3 = b + 1.5 * np.exp(3 * mc) * (b - a)

			r = g[(g > w3)].sort_values(ascending=False)
			one_percent = int(np.round(0.01 * len(g)))
			if len(r) > one_percent:
				r = r.iloc[:one_percent]
			keep_E.loc[keep_E["time"].isin(r.index) & (ccds == ccd), [color, color_err]] = np.nan
	
	# Load and remove JBM times
	logging.info("Removing JBM times")
	discard = pd.read_csv(jbm_discard_path, sep="/", usecols=[0, 1, 2, 3, 5], names=["target", "n_ccd", "n_color", "n_quart", "name"])
	discard["n_time"] = discard["name"].str[10:-5]
	times_translation = pd.read_csv(jbm_date_conversion, usecols=[0, 3], names=["n_time", "hjd"])
	discard = pd.merge(discard, times_translation, on="n_time")
	discard.loc[:, "hjd"] = discard.hjd.astype(float)
	discard = discard[discard["target"]=="lm"].reset_index(drop=True)
	discard.hjd = discard.hjd + 49999.5
	for i, color in enumerate(["red_E", "blue_E"]):
		td = discard[discard.n_color.str[-1]==str(i)]
		keep_E.loc[:, "n_quart"] = keep_E.id_E.str[:5] + str(i) + keep_E.id_E.str[5]
		rm = keep_E[["n_quart", "time"]].isin(discard[["n_quart", "hjd"]]).all(axis=1)
		logging.info("Removed "+str(rm.sum())+" points in "+color)
		keep_E.loc[rm, color] = np.nan
	keep_E.drop("n_quart", axis=1, inplace=True)

	#droping empty lines
	keep_E.dropna(subset=["red_E", "blue_E"], how="all", inplace=True)
	keep_E = pd.merge(keep_E, ids, on="id_E", how="left")

	#LOADING MACHO
	logging.info("Loading MACHO light curves.")
	splitted = ids.id_M.str.split(":", expand=True)
	tiles = splitted[0].str.cat(splitted[1], "_").unique()
	del splitted
	keep_M = []
	for tile in tiles:
		logging.info("   loading tile : "+str(tile))
		try:
			# path = os.path.join("/sps/eros/users/blaineau/macho_fast_read/",
			#                    "F_"+tile.split("_")[0], "F_"+tile+".parquet",
			#                   )
			path = os.path.join(macho_files_path + tile.split("_")[0],
								"F_" + tile.replace("_", ".") + ".gz")
		except TypeError:
			continue
		d = pd.read_csv(path,
						sep=";", usecols=[1, 2, 3, 4, 9, 10, 17, 24, 25, 32],
						names=["field", "tile", "starid", "time", "red_M", "rederr_M", "red_amp", "blue_M", "blueerr_M",
							   "blue_amp"]
						)
		d.loc[:, "id_M"] = d.field.astype(str).str.cat([d.tile.astype(str), d.starid.astype(str)], sep=":")
		d = d.drop(["field", "tile", "starid"], axis=1)
		# d = pd.read_parquet(path)#, columns=["time", "red_M", "rederr_M", "blue_M", "blueerr_M", "id_M"])
		keep_M.append(d[d.id_M.isin(ids.id_M)])
	del d
	keep_M = pd.concat(keep_M)

	# Cleaning MACHO lcs.
	max_macho_fraction = 0.05
	logging.info("Cleaning MACHO light curves")
	fields = keep_M.id_M.str.split(":").str[0]
	for field in fields.unique():
		dfb = pd.DataFrame(np.load(os.path.join(macho_ratio_path, str(field) + "_blue_M_ratios.npy")), columns=["blue_amp", "time", "ratio"])
		pms = list(zip(keep_M["time"].values, keep_M["blue_amp"].values))
		pdf = list(zip(dfb[dfb.ratio > max_macho_fraction]["time"].values, dfb[dfb.ratio > max_macho_fraction]["blue_amp"].values))
		result = pd.Series(pms).isin(pdf) & (fields == field)
		keep_M.loc[result, "blue_M"] = np.nan
		keep_M.loc[result, "blueerr_M"] = np.nan

		dfr = pd.DataFrame(np.load(os.path.join(macho_ratio_path, str(field) + "_red_M_ratios.npy")), columns=["red_amp", "time", "ratio"])
		pms = list(zip(keep_M["time"].values, keep_M["red_amp"].values))
		pdf = list(zip(dfr[dfr.ratio > max_macho_fraction]["time"].values, dfr[dfr.ratio > max_macho_fraction]["red_amp"].values))
		result = pd.Series(pms).isin(pdf) & (fields == field)
		keep_M.loc[result, "red_M"] = np.nan
		keep_M.loc[result, "rederr_M"] = np.nan

	keep_M.drop(["blue_amp", "red_amp"], axis=1, inplace=True)
	keep_M = pd.merge(keep_M, ids, on="id_M", how="left")

	# MERGING LIGHT CURVES
	logging.info("Merging light curves...")
	merged = pd.concat([keep_E, keep_M])
	del keep_E
	del keep_M

	# save merged dataframe
	if save:
		logging.info("Saving")
		merged.to_parquet(os.path.join(output_dir_path, str(start)+"_"+str(end)+".parquet"), index=False)

	logging.info("Total merging time : "+str(time.time()-st1))
	return merged
