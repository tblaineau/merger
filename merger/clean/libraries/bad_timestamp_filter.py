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
	pd.DataFrame.from_dict(lc).to_pickle(os.path.join(output_path, filename[:-3]+".bz2"), compression='bz2')


def MACHO_get_bad_timestamps(field, pickles_path, output_filepath):
	"""
	Save bad timestamp/amp pairs

	Parameters
	----------
	field : int
		MACHO field
	pickles_path : str
		path to the pickle lightcurves
	output_filepath : str
		where to save bad timestamps
	"""
	field = "F_"+str(field)
	lpds = []
	for f in os.listdir(os.path.join(pickles_path, field)):
		lpds.append(pd.read_pickle(os.path.join(field_pickles_path, f)))
	df = pd.concat(lpds)
	del lpds

	#Clean invalid points by replacing bad values by nan. Remove empty rows.
	df = df.replace(
		to_replace={'red_M': -99.,
					'blue_M': -99.},
		value=np.nan)
	df['blueerr_M'] = np.where(df.blueerr_M.between(0, 9.999, inclusive=False), df.blueerr_M, np.nan)
	df['rederr_M'] = np.where(df.rederr_M.between(0, 9.999, inclusive=False), df.rederr_M, np.nan)
	df.dropna(axis='index', how='all', subset=['red_M', 'blue_M'], inplace=True)

	#Compute distance of each points. It is the difference between the median value of the lightcurve and the value of the point divided by the error of the point
	df[['median_red_M', 'median_blue_M']] = df.groupby('id_M')[['red_M', 'blue_M']].transform('median')
	df['red_distance'] = (df['median_red_M'] - df['red_M']) / df['rederr_M']
	df['blue_distance'] = (df['median_blue_M'] - df['blue_M']) / df['blueerr_M']

	red_ratios = df.groupby(['red_amp', 'time'])['red_distance'].agg(lambda x: x[x.abs() > 5].count() / x.count())
	blue_ratios = df.groupby(['blue_amp', 'time'])['blue_distance'].agg(lambda x: x[x.abs() > 5].count() / x.count())

	red_timestamps = red_ratios.index[red_ratios>0.1].tolist()
	blue_timestamps = blue_ratios.index[blue_ratios>0.1].tolist()

	print(red_timestamps)
	print(blue_timestamps)

	np.save(os.path.join(output_filepath, field+"_red_bad_timestamps"), red_timestamps)
	np.save(os.path.join(output_filepath, field+"_blue_bad_timestamps"), blue_timestamps)

MACHO_gz_path = "/Volumes/DisqueSauvegarde/MACHO/lightcurves/F_1"
for f in os.listdir(MACHO_gz_path):
	if f[-3:] == '.gz':
		print(f)
		MACHO_raw_to_pickle(f, MACHO_gz_path, "/Volumes/DisqueSauvegarde/working_dir/pickles/F_1")

MACHO_get_bad_timestamps("/Volumes/DisqueSauvegarde/working_dir/pickles/F_1")