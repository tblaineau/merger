import argparse, time, os
import numpy as np

import sys
sys.path.append('libraries')
import merger_library

if __name__ == '__main__':
	# args = parser = argparse.ArgumentParser()
	# parser.add_argument('MACHO field', metavar='str', type=str)

	### WIP
	
	WORKING_DIR_PATH = "/Volumes/DisqueSauvegarde/working_dir/"
	MACHO_field = 49
	quart = 'lm0324'
	print(quart[:5])


	start = time.time()


	# l o a d   E R O S
	print("Loading EROS files")

	eros_lcs = merger_library.load_eros_files("/Volumes/DisqueSauvegarde/EROS/lightcurves/lm/"+quart[:5]+"/"+quart)

	end_load_eros = time.time()
	print(str(end_load_eros-start)+' seconds elapsed for loading EROS files')

	#loading correspondance file and merging with load EROS stars 
	print("Merging")
	correspondance_path="/Users/tristanblaineau/"+MACHO_field+".txt"
	correspondance = pd.read_csv(correspondance_path, names=["id_E", "id_M"], usecols=[0, 3], sep=' ')
	merged1 = eros_lcs.merge(correspondance, on="id_E", validate="m:1")
	del eros_lcs

	# determine needed tiles from MACHO
	tiles = np.unique([x.split(":")[1] for x in merged1.id_M.unique()])
	if not tiles.size:
		raise NameError("No common stars in field !!!!")

	#l o a d   M A C H O 
	print("Loading MACHO files")
	macho_lcs = merger_library.load_macho_tiles(MACHO_field, tiles)

	merged2 = macho_lcs.merge(correspondance, on='id_M', validate="m:1")
	del macho_lcs
	merged = pd.concat((merged1, merged2), copy=False)

	# replace invalid values in magnitudes with numpy nan
	# and remove rows with no valid magnitude
	merged = merged.replace(to_replace=[99.999,-99.], value=np.nan).dropna(axis=0, how='all', subset=['blue_E', 'red_E', 'blue_M', 'red_M'])

	# remove lightcurves missing one or more color
	merged = merged.groupby('id_E').filter(lambda x: x.red_E.count()!=0 
		and x.red_M.count()!=0 
		and x.blue_E.count()!=0 
		and x.blue_M.count()!=0)

	# save merged dataframe
	merged.to_pickle(os.path.join(WORKING_DIR_PATH, str(MACHO_field)+"_"+quart+".pkl"))