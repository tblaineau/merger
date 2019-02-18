import argparse 
import time

import sys
sys.path.append('/Users/tristanblaineau/Documents/Work/Python/merger/old')
import merger_library

if __name__ == '__main__':
	# args = parser = argparse.ArgumentParser()
	# parser.add_argument('MACHO field', metavar='str', type=str)

	### WIP
	
	WORKING_DIR_PATH = "/Volumes/DisqueSauvegarde/working_dir/"
	MACHO_field = 49
	quart = 'lm0324'
	print(quart[:4])


	start = time.time()


	# l o a d   E R O S
	print("Loading EROS files")

	eros_lcs = merger_library.load_eros_files("/Volumes/DisqueSauvegarde/EROS/lightcurves/lm/"+quart[:4]+"/lm0322")

	end_load_eros = time.time()
	print(str(end_load_eros-start)+' seconds elapsed for loading EROS files')

	print("Merging")
	# 
	correspondance_path="/Users/tristanblaineau/"+MACHO_field+".txt"
	correspondance = pd.read_csv(correspondance_path, names=["id_E", "id_M"], usecols=[0, 3], sep=' ')
	merged1 = eros_lcs.merge(correspondance, on="id_E", validate="m:1")
	del eros_lcs
	tiles = np.unique([x.split(":")[1] for x in merged1.id_M.unique()])
	if not tiles.size:
		raise NameError("No common stars in field !!!!")

	#l o a d   M A C H O 
	print("Loading MACHO files")
	macho_lcs = load_macho_tiles(MACHO_field, tiles)

	merged2 = macho_lcs.merge(correspondance, on='id_M', validate="m:1")
	del macho_lcs
	merged = pd.concat((merged1, merged2), copy=False)
