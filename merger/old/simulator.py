import pandas as pd
import numpy as np

from merger.clean.libraries.merger_library import COLOR_FILTERS
from merger.clean.libraries.parameter_generator import Microlensing_generator, microlens_parallax, microlens_simple


def load_merged(filepath, fraction=1):
	"""
	Load file of merged lightcurves

	Parameters
	----------
	filepath : str
		Path to a pandas pickle file containing the lightcurves
	fraction : float
		Fraction of the lightcurves to keep (for simulation)

	Returns
	-------
	pd.DataFrame
		fraction of the original merged lightcurves
	"""
	df = pd.read_pickle(filepath)
	if fraction != 1:
		df = df.groupby('id_M').filter(lambda x: np.random.uniform(0,1) < fraction)
	return df


def clean_lightcurves(df):
	"""Clean dataframe from bad errors."""

	# magnitude should have already been cleaned but we never know
	df = df.replace(to_replace=[99.999, -99.], value=np.nan).dropna(axis=0, how='all', subset=list(COLOR_FILTERS.keys()))
	for c in COLOR_FILTERS.values():
		df.loc[~df[c['err']].between(0, 9.999, inclusive=False), [c['mag'], c['err']]] = np.nan
	#drop line without information
	df.dropna(how='all', subset=list(COLOR_FILTERS.keys()), inplace=True)
	return df


class Sigmag_histogram:
	#TODO : save and load mechanism
	def __init__(self, df):
		self.bin_edges = {}
		self.bin_values = {}
		for c in COLOR_FILTERS.values():
			df['bins_'+c['mag']], self.bin_edges[c['mag']] = pd.qcut(df[c['mag']], 25, retbins=True)
			self.bin_values[c['mag']] = df.groupby('bins_'+c['mag'])[c['err']].median()

	def get_sigma(self, cfilter, mag):
		"""Get median sigma"""
		index = np.digitize(mag, self.bin_edges[cfilter])
		index = np.where(index>=len(self.bin_edges[cfilter])-2, len(self.bin_values[cfilter])-1, index-1)
		return self.bin_values[cfilter][index]


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
	print(current_id)

	#Generate µlens parameters
	params = generator.generate_parameters(current_id)
	u0, t0, tE, blend_factors, delta_u, theta = params["u0"], params["t0"], params["tE"], params["blend"], params["delta_u"], params["theta"]

	time = subdf.time.values

	for key, color_filter in COLOR_FILTERS.items():
		mag_i = subdf[color_filter['mag']].values
		mag_med = np.nanmedian(mag_i)
		if parallax:
			mag_th = microlens_parallax(time, mag_med, blend_factors[key], u0, t0, tE, delta_u, theta)
		else:
			mag_th = microlens_simple(time, mag_med, blend_factors[key], u0, t0, tE, delta_u, theta)
		#mag_th is the lightcurve with perfect measurements (computed from original baseline)

		norm = sigmag.get_sigma_base(mag_med) / sigmag.get_sigma_base(mag_th)

		subdf['ampli_'+color_filter['err']] = subdf[color_filter['err']].values / norm
		subdf['ampli_'+color_filter['mag']] = mag_th + (mag_i - mag_med) / norm

	return subdf





"""WORKING_DIR_PATH = "/Volumes/DisqueSauvegarde/working_dir/"

# l o a d   s t a r s
print("Loading stars")
merged = pd.read_pickle(WORKING_DIR_PATH+"50_lm0220.pkl")

print(str(len(merged))+" lines loaded.")

# l o a d   b a s e l i n e s   a n d   s t d   d e v i a t i o n s 
print("Loading mag and sig")
ms = pd.read_pickle(WORKING_DIR_PATH+"ms_temp_lm0322n")

# c o m p u t e   b i n n e d   h i s t o g r a m   o f   s t d   d e v i a t i o n s
bin_baseline, bin_edges, binnumber = stats.binned_statistic(ms.dropna(subset=['bl_red_E', 'std_red_E']).bl_red_E, ms.dropna(subset=['bl_red_E', 'std_red_E']).std_red_E, bins=30, statistic='mean')
sigmag = Sigma_Baseline(bin_edges, bin_baseline)

#delete stars with at least one empty lightcurves
merged = merged.groupby('id_E').filter(lambda x: x.red_E.count()!=0 
	and x.red_M.count()!=0 
	and x.blue_E.count()!=0 
	and x.blue_M.count()!=0)

#simulate on only x% of the lightcurves
merged = merged.groupby("id_E").filter(lambda x: np.random.rand()<0.02)

g = Microlensing_generator(seed=12345)

print("Starting simulations...")
start = time.time()
simulated = merged.groupby("id_E").apply(generate_microlensing_events, sigmag=sigmag, raw_stats_df=ms, generator=g, parallax=True)
print(time.time()-start)
#simulated.to_pickle(WORKING_DIR_PATH+'simulated_50_lm0220_parallax.pkl')"""