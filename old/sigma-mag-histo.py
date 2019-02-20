import numpy as np
import pandas as pd
import os, gzip
from scipy.optimize import curve_fit
import scipy.stats as stats
from iminuit import Minuit
import matplotlib.pyplot as plt
import seaborn as sns
import time
from functools import reduce

from merger_library import *

### CONVERT TO PANDAS PICKLE
# field_path = "/Volumes/DisqueSauvegarde/EROS/lightcurves/lm/lm032/"
# for dirs in os.listdir(field_path):
# 	print(dirs)
# 	if os.path.isdir(os.path.join(field_path,dirs)) and not (dirs=="lm0320"):
# 		for subdir in os.listdir(os.path.join(field_path,dirs)):
# 			print(subdir)
# 			if os.path.isdir(os.path.join(field_path,dirs, subdir)):
# 				full_field = open_eros_files(os.path.join(field_path,dirs,subdir))
# 				full_field.to_pickle("/Volumes/DisqueSauvegarde/working_dir/full_"+subdir)
# 				del full_field


NB_POINTS=1000
SCALE_WIDTH = 3

def gaussian(x, mu, sig, A):
	return A/np.sqrt(2*np.pi*sig**2)*np.exp(-(x-mu)**2/(2*sig**2))

def compute_magstats(df):
	if len(df)==0:
		return np.nan, np.nan
	width = df.mag.max()-df.mag.min()
	x=np.linspace(df.mag.min()-width, df.mag.max()+width, NB_POINTS)
	y=np.zeros(NB_POINTS)
	for r, e in zip(df.mag, df.err):
		y+=stats.norm.pdf(x, loc=r, scale=SCALE_WIDTH*e)


	def minimize_gaussian(mu, sigma, A):
		return np.sum((gaussian(x, mu, sigma, A)-y)**2)
	
	m = Minuit(minimize_gaussian, 
				mu=df.mag.mean(), 
				sigma=1, 
				A=1000, error_mu=1, error_sigma=0.5, error_A=100, errordef=1, print_level=0)
	fmin, param = m.migrad()
	if m.get_fmin().is_valid:
		return m.values['mu'], np.sum((df.mag-m.values['mu'])**2/len(df.mag))
	else:
		return np.nan, np.nan

def mag_stats(df):
	""" Take df with columns names [id_E, mag, err] and return baseline and std deviation"""
	# ms = []
	# counter=0
	# for name, group in df.groupby("id_E"):
	# 	line = [name]
	# 	counter+=1
	# 	print(str(counter)+" : " +name, end="\r")
	# 	for key, cf in COLOR_FILTERS.items():
	# 		c = group[cf['mag']].notnull() & (group[cf['err']]>0) & (group[cf['err']]<9.999)
	# 		line+=compute_magstats(group[c][[cf['mag'], cf['err']]].rename(columns={cf['mag']:'mag', cf['err']:'err'}))
	# 	ms.append(line)

	# ms = pd.DataFrame(ms)
	# col_names = ['id_E']
	# for key, value in COLOR_FILTERS.items():
	# 	col_names.append('bl_'+key)
	# 	col_names.append('std_'+key)
	# ms.columns = col_names
	# return ms
	
	stats_list = []
	for key, cf in COLOR_FILTERS.items():
		c = df[cf['mag']].notnull() & (df[cf['err']]>0) & (df[cf['err']]<9.999)
		stats_list.append(df[c].groupby('id_E')[cf['mag']].agg(['mean', 'std']))
	ms = stats_list[0]
	for e in stats_list[1:]:
		ms = ms.merge(e, on='id_E')

	col_names = []
	for key, value in COLOR_FILTERS.items():
		col_names.append('bl_'+key)
		col_names.append('std_'+key)
	print(ms.columns)
	ms.columns = col_names
	return ms


###################### LOAD TEST DATA

# ef = pd.read_pickle("/Volumes/DisqueSauvegarde/working_dir/full_lm0322n")
# ef = ef.replace(to_replace=99.999, value=np.nan).dropna(axis=0, how='all', subset=['blue_E', 'red_E'])
# start = time.time()
# mf = load_macho_tiles(49, [6382])
# # mf["id_M"] = mf[["id1", "id2", "id3"]].apply(lambda x: ':'.join([str(i) for i in x]), axis=1)
# # mf.drop(["id1", "id2", "id3"], axis=1, inplace=True)
# mf = mf.replace(to_replace=-99., value=np.nan).dropna(axis=0, how='all', subset=['blue_M', 'red_M'])
# print(str(time.time()-start)+" seconds to load MACHO data.")
# correspondance_path="/Users/tristanblaineau/49.txt"
# correspondance = pd.read_csv(correspondance_path, names=["id_E", "id_M"], usecols=[0, 3], sep=' ')
# merged1 = ef.merge(correspondance, on="id_E", validate="m:1")
# del ef
# merged2 = mf.merge(correspondance, on='id_M', validate="m:1")
# del mf
# merged = pd.concat((merged1, merged2))


WORKING_DIR_PATH = "/Volumes/DisqueSauvegarde/working_dir/"

merged=pd.read_pickle(WORKING_DIR_PATH+"49_lm0324.pkl")

merged = merged.groupby('id_E').filter(lambda x: x.red_E.count()!=0 and x.blue_E.count()!=0 and x.red_M.count()!=0 and x.blue_M.count()!=0)

start = time.time()
ms = mag_stats(merged)
print(str(time.time()-start)+" seconds elapsed.")

ms.to_pickle('ms_temp_lm0324')