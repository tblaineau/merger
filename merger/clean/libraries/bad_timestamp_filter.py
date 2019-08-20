import numpy as np
import gzip
import os
import logging
import pandas as pd

def MACHO_raw_to_pickle(filename, input_path, output_path):
	"""
	Read MACHO gzipped lightcurve file and save it to pickle

	Parameters
	----------
	filename : str
		Name of the file to read (gzip)
	input_path : str
		Path of the input file directory
	output_path : str
		Path where to save the resulting pickle file
	"""
	try:
		with gzip.open(os.path.join(input_path, filename), 'rt') as f:
			lc = {'time': [],
				  'red_M': [],
				  'rederr_M': [],
				  'blue_M': [],
				  'blueerr_M': [],
				  'id_M': [],
				  'red_amp': [],
				  'blue_amp': [],
				  'observation_id': []
				  }
			for line in f:
				line = line.split(';')
				lc['id_M'].append(line[1] + ":" + line[2] + ":" + line[3])
				lc['time'].append(float(line[4]))
				lc['red_M'].append(float(line[9]))
				lc['rederr_M'].append(float(line[10]))
				lc['blue_M'].append(float(line[24]))
				lc['blueerr_M'].append(float(line[25]))
				lc['red_amp'].append(float(line[17]))
				lc['blue_amp'].append(float(line[32]))
				lc['observation_id'].append(int(line[5]))
	except FileNotFoundError:
		logging.error(f"File {os.path.join(input_path, filename)} not found.")
	pd.DataFrame.from_dict(lc).to_pickle(os.path.join(output_path, filename[:-3]+".pkl"))

#MACHO_raw_to_pickle("F_1.3319.gz", "/Volumes/DisqueSauvegarde/MACHO/lightcurves/F_1", "/Users/tristanblaineau/Desktop")