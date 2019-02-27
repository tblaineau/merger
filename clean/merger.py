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

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--EROS-field', '-fE', type=str, required=True)
	parser.add_argument('--EROS-CCD', '-ccdE', type=int, default=None, choices=np.arange(7), required=False)
	parser.add_argument('--MACHO-field', '-fM', type=int, required=True)
	parser.add_argument('--output-directory', '-odir', type=str, default=merger_library.WORKING_DIR_PATH)
	parser.add_argument('-fit', action='store_true')

	args = parser.parse_args()

	if os.path.isdir(args.output_directory):
		working_dir_path = args.output_directory
	else:
		raise Exception("This directory doesn't exist : "+args.output_directory)

	MACHO_field = args.MACHO_field
	EROS_field = args.EROS_field
	EROS_CCD = args.EROS_CCD
	fit = args.fit

	print(fit)

	if EROS_CCD:
		eros_ccd = "lm"+EROS_field+str(EROS_CCD)
		merger_library.merger(working_dir_path, MACHO_field, eros_ccd)
		if fit:
			iminuit_fitter.fit_all(str(MACHO_field)+"_"+str(eros_ccd)+".pkl")
	else:
		for i in range(0,8):
			eros_ccd = "lm"+EROS_field+str(i)
			merger_library.merger(working_dir_path, MACHO_field, eros_ccd)
			if fit:
				iminuit_fitter.fit_all(str(MACHO_field)+"_"+str(eros_ccd)+".pkl")