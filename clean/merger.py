#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to load lightcuvres from EROS and MACHO database and save the merged result

Load lightcurves from EROS and MACHO databases, load association file, that should be already computed, merge the lightcurves and save it in a pandas pickle file
"""

import argparse, time, os
import numpy as np
import pandas as pd
import logging

import sys
sys.path.append('libraries')
import merger_library

if __name__ == '__main__':
	# args = argparse.ArgumentParser()
	# args.add_argument(['--EROS-field', '-fE'], metavar='str', type=str)
	# args.add_argument(['--EROS-CCD', '-ccdE'], metavar='int', type=int, default=None, choices=np.arange(7), required=False)
	### WIP
	
	WORKING_DIR_PATH = "/Volumes/DisqueSauvegarde/working_dir/"
	MACHO_field = 49
	ccd = 'lm0325'


	start = time.time()


	# l o a d   E R O S
	logging.info("Loading EROS files")


	# eros_lcs = pd.concat([pd.read_pickle(WORKING_DIR_PATH+"full_"+ccd+quart) for quart in 'klmn'])				# <===== Load from pickle files
	eros_lcs = merger_library.load_eros_files("/Volumes/DisqueSauvegarde/EROS/lightcurves/lm/"+ccd[:5]+"/"+ccd)
	end_load_eros = time.time()
	logging.info(str(end_load_eros-start)+' seconds elapsed for loading EROS files')

	#loading correspondance file and merging with load EROS stars 
	logging.info("Merging")
	correspondance_path="/Users/tristanblaineau/"+str(MACHO_field)+".txt"
	correspondance = pd.read_csv(correspondance_path, names=["id_E", "id_M"], usecols=[0, 3], sep=' ')
	merged1 = eros_lcs.merge(correspondance, on="id_E", validate="m:1")
	del eros_lcs

	# determine needed tiles from MACHO
	tiles = np.unique([x.split(":")[1] for x in merged1.id_M.unique()])
	if not tiles.size:
		raise NameError("No common stars in field !!!!")

	#l o a d   M A C H O 
	logging.info("Loading MACHO files")
	macho_lcs = merger_library.load_macho_tiles(MACHO_field, tiles)

	logging.info("Merging")
	merged2 = macho_lcs.merge(correspondance, on='id_M', validate="m:1")
	del macho_lcs
	merged = pd.concat((merged1, merged2), copy=False)

	# replace invalid values in magnitudes with numpy NaN and remove rows with no valid magnitudes
	merged = merged.replace(to_replace=[99.999,-99.], value=np.nan).dropna(axis=0, how='all', subset=['blue_E', 'red_E', 'blue_M', 'red_M'])

	# remove lightcurves missing one or more color
	merged = merged.groupby('id_E').filter(lambda x: x.red_E.count()!=0 
		and x.red_M.count()!=0 
		and x.blue_E.count()!=0 
		and x.blue_M.count()!=0)

	# save merged dataframe
	logging.info("Saving")
	merged.to_pickle(os.path.join(WORKING_DIR_PATH, str(MACHO_field)+"_"+ccd+".pkl"))