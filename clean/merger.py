#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to load lightcuvres from EROS and MACHO database and save the merged result

Load lightcurves from EROS and MACHO databases, load association file, that should be already computed, merge the lightcurves and save it in a pandas pickle file
"""

import argparse
import numpy as np
import logging
import os

import sys
sys.path.append('libraries')
import merger_library, iminuit_fitter

def dir_path_check(dirpath):
	if not os.path.isdir(dirpath):
		raise Exception("This directory doesn't exist : "+dirpath)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--EROS-field', '-fE', type=str, required=True)
	parser.add_argument('--EROS-CCD', '-ccdE', type=int, default=None, choices=np.arange(7), required=False)
	parser.add_argument('--MACHO-field', '-fM', type=int, required=True)
	parser.add_argument('--output-directory', '-odir', type=str, default=merger_library.OUTPUT_DIR_PATH)
	parser.add_argument('-fit', action='store_true')
	parser.add_argument('--EROS-path', '-pE', type=str, default="/Volumes/DisqueSauvegarde/EROS/lightcurves/lm/")
	parser.add_argument('--MACHO-path', '-pM', type=str, default="/Volumes/DisqueSauvegarde/MACHO/lightcurves/")
	parser.add_argument('--correspondance-path', '-pC', type=str, default="/Users/tristanblaineau/")

	args = parser.parse_args()

	MACHO_field = args.MACHO_field
	EROS_field = args.EROS_field
	EROS_CCD = args.EROS_CCD
	fit = args.fit
	EROS_files_path = args.EROS_path
	MACHO_files_path = args.MACHO_path
	correspondance_files_path = args.correspondance_path
	output_directory = args.output_directory

	dir_path_check(output_directory)
	dir_path_check(correspondance_files_path)
	dir_path_check(MACHO_files_path)
	dir_path_check(EROS_files_path)

	print(fit)

	if EROS_CCD:
		eros_ccd = "lm"+EROS_field+str(EROS_CCD)
		merger_library.merger(working_dir_path, MACHO_field, eros_ccd, EROS_files_path, correspondance_files_path, MACHO_files_path)
		if fit:
			iminuit_fitter.fit_all(str(MACHO_field)+"_"+str(eros_ccd)+".pkl")
	else:
		for i in range(0,8):
			eros_ccd = "lm"+EROS_field+str(i)
			merger_library.merger(working_dir_path, MACHO_field, eros_ccd, EROS_files_path, correspondance_files_path, MACHO_files_path)
			if fit:
				iminuit_fitter.fit_all(str(MACHO_field)+"_"+str(eros_ccd)+".pkl")