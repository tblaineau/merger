import numpy as np
import pandas as pd
import scipy.stats as stats
import time

from merger.clean.libraries.merger_library import *
from merger.clean.libraries.parameter_generator import Microlensing_generator

def microlensing_amplification(t, u0, t0, tE):
	u = np.sqrt(u0*u0 + ((t-t0)**2)/tE/tE)
	return (u**2+2)/(u*np.sqrt(u**2+4))

def ma_parallax(t, u0, t0, tE, delta_u, theta):
	year = 365.2422
	alphaS = 80.8941667*np.pi/180.
	deltaS = -69.7561111*np.pi/180.
	epsilon = (90. - 66.56070833)*np.pi/180.
	t_origin = 58747 #(21 septembre 2019)
	sin_beta = np.cos(epsilon)*np.sin(deltaS) - np.sin(epsilon)*np.cos(deltaS)*np.sin(alphaS)
	beta = np.arcsin(sin_beta) #ok because beta is in [-pi/2; pi/2]
	if abs(beta)==np.pi/2:
		lambda_star = 0
	else:
		lambda_star = np.sign((np.sin(epsilon)*np.sin(deltaS)+np.cos(epsilon)*np.sin(alphaS)*np.cos(deltaS))/np.cos(beta)) * np.arccos(np.cos(deltaS)*np.cos(alphaS)/np.cos(beta))
	tau = (t-t0)/tE
	phi = 2*np.pi * (t-t_origin)/year - lambda_star
	u_D = np.array([ 
		-u0*np.sin(theta) + tau*np.cos(theta),
		 u0*np.cos(theta) + tau*np.sin(theta)
		])
	u_t = np.array([
		-delta_u*np.sin(phi),
		 delta_u*np.cos(phi)*sin_beta
		])
	u = np.linalg.norm(u_D-u_t, axis=0)
	return (u**2+2)/(u*np.sqrt(u**2+4))

class Sigma_Baseline:
	def __init__(self, bin_edges, bin_baseline):
		self.be = bin_edges
		self.bb = bin_baseline
	def get_sigma_base(self,mag):
		index = np.digitize(mag, self.be)
		index = np.where(index>=len(self.be)-2, len(self.bb)-1, index-1)
		return self.bb[index]

def generate_microlensing_events(subdf, sigmag, raw_stats_df, generator, parallax=False):
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

	#Remove anomalous values of magnitudes
	subdf=subdf.replace(to_replace=[99.999,-99.], value=np.nan).dropna(axis=0, how='all', subset=['blue_E', 'red_E', 'blue_M', 'red_M'])

	current_id = subdf["id_E"].iloc[0]
	print(current_id)

	means = {}
	conditions = {}
	for key, color_filter in COLOR_FILTERS.items():
		conditions[key] = subdf[color_filter['mag']].notnull() & (subdf[color_filter['err']]<9.999) & (subdf[color_filter['err']]>0)
		# means[key] = raw_stats_df.loc[current_id][color_filter['mag']]	# <---- use baseline from "raw stats"
		means[key] = subdf[color_filter['mag']].mean() # <---- use mean as baseline

	#Generate µlens parameters
	params = generator.generate_parameters(current_id)
	u0, t0, tE, blend_factors, delta_u, theta = params["u0"], params["t0"], params["tE"], params["blend"], params["delta_u"], params["theta"]
	if parallax:
		A = ma_parallax(subdf.time, u0, t0, tE, delta_u, theta)
	else:
		A = microlensing_amplification(subdf.time, u0, t0, tE)


	for key, color_filter in COLOR_FILTERS.items():
		#phi_th is the lightcurve with perfect measurements (computed from original baseline)
		phi_th = -2.5*np.log10(np.power(10, means[key]/-2.5)*blend_factors[key] + np.power(10, means[key]/-2.5)*(1-blend_factors[key])*A[conditions[key]])
		norm = sigmag.get_sigma_base(means[key]) / sigmag.get_sigma_base(phi_th)

		subdf['ampli_'+color_filter['err']] = subdf[conditions[key]][color_filter['err']] / norm
		subdf['ampli_'+color_filter['mag']] = phi_th + (subdf[conditions[key]][color_filter['mag']] - means[key]) / norm

	return subdf

WORKING_DIR_PATH = "/Volumes/DisqueSauvegarde/working_dir/"

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
#simulated.to_pickle(WORKING_DIR_PATH+'simulated_50_lm0220_parallax.pkl')