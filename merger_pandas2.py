import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

import time

from merger_library import *

NB_POINTS=1000
RANGE = (-6.5, -2)
SCALE_WIDTH = 3

def gaussian(x, mu, sig, A):
	return A/np.sqrt(2*np.pi*sig**2)*np.exp(-(x-mu)**2/(2*sig**2))

def flat_curve(time, mean_mag):
	return [mean_mag]*len(time)

def microlensing_amplification(t, u0, t0, tE):
	u = np.sqrt(u0*u0 + ((t-t0)**2)/tE/tE)
	return (u**2+2)/(u*np.sqrt(u**2+4))


def moving_window_smoothing(time, mags, width=5):
	""" data = [times, values] """
	smoothed = []
	data=np.array([time, mags])
	for i in range(data.shape[1]-width):
		smoothed.append([data[0, i+width%2], np.median(data[1,i:i+width])])
	return np.array(smoothed).transpose()

class Sigma_Baseline:
	def __init__(self, bin_edges, bin_baseline):
		self.be = bin_edges
		self.bb = bin_baseline
	def get_sigma_base(self,mag):
		index = np.digitize(mag, self.be)
		index = np.where(index>=len(self.be)-2, len(self.bb)-1, index-1)
		return self.bb[index]

def generate_microlensing_events(subdf, sigmag, raw_stats_df):
	subdf=subdf.replace(to_replace=[99.999,-99.], value=np.nan).dropna(axis=0, how='all', subset=['blue_E', 'red_E', 'blue_M', 'red_M'])

	current_id = subdf.id_E.iloc[0]

	means = {}
	conditions = {}
	for key, color_filter in color_filters.items():
		conditions[key] = subdf[color_filter['mag']].notnull() & (subdf[color_filter['err']]<9.999) & (subdf[color_filter['err']]>0)
		# means[key] = raw_stats_df.loc[current_id][color_filter['mag']]	# <--- use baseline
		means[key] = subdf[color_filter['mag']].mean() # <---- use mean as baseline

	#amplification mulitplier
	A = microlensing_amplification(subdf.time, *generate_microlensing_parameters(current_id))
	
	for key, color_filter in color_filters.items():
		phi_th = means[key] - 2.5*np.log10(A[conditions[key]])
		norm = sigmag.get_sigma_base(subdf[conditions[key]][color_filter['mag']]) / sigmag.get_sigma_base(phi_th)

		subdf['ampli_'+color_filter['err']] = subdf[conditions[key]][color_filter['err']] / norm
		subdf['ampli_'+color_filter['mag']] = phi_th + (subdf[conditions[key]][color_filter['mag']] - means[key]) / norm

	return subdf

def visualize_lightcurve(df, id_E):
	## A développer
	temp = df[df["id_E"]==id_E]
	print(temp)
	plt.plot(temp.time, temp.red_E)
	plt.plot(temp.time, temp.blue_E)
	plt.show()



WORKING_DIR_PATH = "/Volumes/DisqueSauvegarde/working_dir/"

start = time.time()
quart = 'lm0324'
# l o a d   E R O S
pds = []
for file in ['full_'+quart+x for x in ['k', 'l', 'm', 'n']]:
	if file[:4]=="full":
		print(file)
		pds.append(pd.read_pickle(os.path.join(WORKING_DIR_PATH+file)))
eros_lcs = pd.concat(pds)
del pds


# c o n v e r t   f r o m   f i e l d : t i l e : s t a r   t o   i d _ M 
# print("Formating id_M...")
# macho_lcs["id_M"] = macho_lcs[["id1", "id2", "id3"]].apply(lambda x: ':'.join([str(i) for i in x]), axis=1)
# macho_lcs.drop(["id1", "id2", "id3"], axis=1, inplace=True)

# l o a d   c o r r e s p o n d a n c e   a n d   m e r g e
print("Merging")
correspondance_path="/Users/tristanblaineau/49.txt"
correspondance = pd.read_csv(correspondance_path, names=["id_E", "id_M"], usecols=[0, 3], sep=' ')
merged1 = eros_lcs.merge(correspondance, on="id_E", validate="m:1")
del eros_lcs
tiles = np.unique([x.split(":")[1] for x in merged1.id_M.unique()])
if not tiles.size:
	print("No common stars in field !!!!")

#l o a d   M A C H O 
print("Loading MACHO files")
macho_lcs = load_macho_tiles(49, tiles)

merged2 = macho_lcs.merge(correspondance, on='id_M', validate="m:1")
del macho_lcs
merged = pd.concat((merged1, merged2), copy=False)

# D e l e t e   l c s   m i s s i n g   a t   l e a s t   o n e   c o l o r   a n d   s a v e 
# print("Filtering and saving")
# merged.groupby('id_E').filter(lambda x: x.red_E.count()!=0 
# 	and x.red_M.count()!=0 
# 	and x.blue_E.count()!=0 
# 	and x.blue_M.count()!=0).to_pickle(WORKING_DIR_PATH+"merged_49_"+quart+".pkl")
# print(time.time() - start)
# merged.to_pickle(WORKING_DIR_PATH+"merged_49_"+quart+".pkl")

# # l o a d   b a s e l i n e s   a n d   s t d   d e v i a t i o n s 
ms = pd.read_pickle("ms_temp_lm0322n")
# c o m p u t e   b i n n e d   h i s t o g r a m   o f   s t d   d e v i a t i o n s
bin_baseline, bin_edges, binnumber = stats.binned_statistic(ms.dropna(subset=['bl_red_E', 'std_red_E']).bl_red_E, ms.dropna(subset=['bl_red_E', 'std_red_E']).std_red_E, bins=30, statistic='mean')
sigmag = Sigma_Baseline(bin_edges, bin_baseline)

merged = merged.groupby('id_E').filter(lambda x: x.red_E.count()!=0 
	and x.red_M.count()!=0 
	and x.blue_E.count()!=0 
	and x.blue_M.count()!=0)
print(merged.id_E.nunique())
merged = merged.groupby("id_E").filter(lambda x: np.random.rand()<0.02)
print(merged.id_E.nunique())
simulated = merged.groupby("id_M").apply(generate_microlensing_events, sigmag=sigmag, raw_stats_df=ms)
simulated.to_pickle('simulated_test.pkl')
