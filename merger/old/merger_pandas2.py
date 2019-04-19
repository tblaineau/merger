import numpy as np
import pandas as pd
import os
import time
from merger_library import *

def gaussian(x, mu, sig, A):
	return A/np.sqrt(2*np.pi*sig**2)*np.exp(-(x-mu)**2/(2*sig**2))

def flat_curve(time, mean_mag):
	return [mean_mag]*len(time)

def moving_window_smoothing(time, mags, width=5):
	""" data = [times, values] """
	smoothed = []
	data=np.array([time, mags])
	for i in range(data.shape[1]-width):
		smoothed.append([data[0, i+width%2], np.median(data[1,i:i+width])])
	return np.array(smoothed).transpose()

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
