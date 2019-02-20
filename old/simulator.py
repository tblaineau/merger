import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
import scipy.stats as stats
import time

from merger_library import *

def microlensing_amplification(t, u0, t0, tE):
	u = np.sqrt(u0*u0 + ((t-t0)**2)/tE/tE)
	return (u**2+2)/(u*np.sqrt(u**2+4))

class Sigma_Baseline:
	def __init__(self, bin_edges, bin_baseline):
		self.be = bin_edges
		self.bb = bin_baseline
	def get_sigma_base(self,mag):
		index = np.digitize(mag, self.be)
		index = np.where(index>=len(self.be)-2, len(self.bb)-1, index-1)
		return self.bb[index]

def generate_microlensing_events(subdf, sigmag, raw_stats_df, blending=False):
	""" For a given star, generate µ-lens event
	
	[description]
	
	Arguments:
		subdf {dataframe} -- Dataframe containing only one star
		sigmag {Sigma_Baseline} -- object containg informations on distribution of standard deviation in function of magnitude, for all filters
		raw_stats_df {dataframe} -- Dataframe containing all means and std deviation of the catalogue
	
	Keyword Arguments:
		blending {bool} -- Simulate blending or not (default: {False})
	
	Returns:
		DataFrame -- New star with a µlens event
	"""
	subdf=subdf.replace(to_replace=[99.999,-99.], value=np.nan).dropna(axis=0, how='all', subset=['blue_E', 'red_E', 'blue_M', 'red_M'])

	current_id = subdf.id_E.iloc[0]

	means = {}
	conditions = {}
	for key, color_filter in COLOR_FILTERS.items():
		conditions[key] = subdf[color_filter['mag']].notnull() & (subdf[color_filter['err']]<9.999) & (subdf[color_filter['err']]>0)
		# means[key] = raw_stats_df.loc[current_id][color_filter['mag']]	# <---- use baseline from "raw stats"
		means[key] = subdf[color_filter['mag']].mean() # <---- use mean as baseline

	#amplification mulitplier
	u0, t0, tE, blend_factors = generate_microlensing_parameters(current_id, blending=blending)
	A = microlensing_amplification(subdf.time, u0, t0, tE)

	
	for key, color_filter in COLOR_FILTERS.items():
		#phi_th is the lightcurve with perfect measurements (computed from original baseline)
		#phi_th = means[key] - 2.5*np.log10(A[conditions[key]])
		phi_th = -2.5*np.log10(np.power(10, means[key]/-2.5)*blend_factors[key] + np.power(10, means[key]/-2.5)*(1-blend_factors[key])*A[conditions[key]])
		norm = sigmag.get_sigma_base(subdf[conditions[key]][color_filter['mag']]) / sigmag.get_sigma_base(phi_th)

		subdf['ampli_'+color_filter['err']] = subdf[conditions[key]][color_filter['err']] / norm
		subdf['ampli_'+color_filter['mag']] = phi_th + (subdf[conditions[key]][color_filter['mag']] - means[key]) / norm

	return subdf

# l o a d   s t a r s
print("Loading stars")
merged = pd.read_pickle(WORKING_DIR_PATH+"49_lm0322.pkl")

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

#simulate on only 2% of the lightcurves
merged = merged.groupby("id_E").filter(lambda x: np.random.rand()<0.02)

print("Starting simulations...")
start = time.time()
simulated = merged.groupby("id_E").apply(generate_microlensing_events, sigmag=sigmag, raw_stats_df=ms, blending=True)
print(time.time()-start)
simulated.to_pickle('simulated_test_2.pkl')