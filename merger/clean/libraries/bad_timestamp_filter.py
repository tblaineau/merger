import numpy as np
import gzip
import os
import logging
import pandas as pd

from irods.session import iRODSSession
from irods.exception import CollectionDoesNotExist, DataObjectDoesNotExist
import ssl
import tarfile

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


def read_macho_lightcurve(filepath):
	try:
		with gzip.open(filepath, 'rt') as f:
			lc = {'time':[],
				  'red_M':[],
				  'rederr_M':[],
				  'blue_M':[],
				  'blueerr_M':[],
				  'id_M':[],
				  'red_amp': [],
				  'blue_amp': [],
				  "observation_id":[]}
			for line in f:
				line = line.split(';')
				lc['id_M'].append(line[1]+":"+line[2]+":"+line[3])
				lc['time'].append(float(line[4]))
				lc['red_M'].append(float(line[9]))
				lc['rederr_M'].append(float(line[10]))
				lc['blue_M'].append(float(line[24]))
				lc['blueerr_M'].append(float(line[25]))
				lc['red_amp'].append(float(line[17]))
				lc['blue_amp'].append(float(line[32]))
				lc['observation_id'].append(int(line[5]))
			f.close()
	except FileNotFoundError:
		print(filepath+" doesn't exist.")
		return None
	return pd.DataFrame.from_dict(lc)


def MACHO_get_bad_timestamps(field, output_path, pickles_path=None, archives_path=None):
	"""
	Save ratio of bad points over all points of each amp and each time.

	Parameters
	----------
	field : int
		MACHO field
	pickles_path : str
		path to the pickle lightcurves (can be merged lightcurves)
	output_path : str
		where to save bad timestamps
	archives_path : str
		path to the .gz macho lightcurves

	"""
	lpds = []
	if pickles_path:
		field_pickles_path = os.path.join(pickles_path, "F_"+str(field))
		for f in os.listdir(field_pickles_path):
			lpds.append(pd.read_pickle(os.path.join(field_pickles_path, f)))
	elif archives_path:
		field_archives_path = os.path.join(archives_path, "F_" + str(field))
		for f in os.listdir(field_archives_path):
			lpds.append(read_macho_lightcurve(os.path.join(field_archives_path, f)))
	else:
		logging.ERROR('No import path.')
		return 1
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
	#TODO : Compare to sigma intrinsic rather than vanilla errors !
	df['red_distance'] = (df['median_red_M'] - df['red_M']) / df['rederr_M']
	df['blue_distance'] = (df['median_blue_M'] - df['blue_M']) / df['blueerr_M']

	red_ratios = df.groupby(['red_amp', 'time'])['red_distance'].agg(lambda x: x[x.abs() > 5].count() / x.count())
	blue_ratios = df.groupby(['blue_amp', 'time'])['blue_distance'].agg(lambda x: x[x.abs() > 5].count() / x.count())

	np.save(os.path.join(output_path, str(field)+"_red_M_ratios"), red_ratios)
	np.save(os.path.join(output_path, str(field)+"_blue_M_ratios"), blue_ratios)
	return 0

def EROS_get_bad_timestamp(irods_path, output_path):
	#TODO: write the function
	"""
	Get lightcurves from one cdd of one EROS field

	Parameters
	----------
	irods_path : str
		Path to irods ccd directory containting the .tar.gz
	output_path : str
		Where to save the timestamps
	"""

	try:
		env_file = os.environ['IRODS_ENVIRONMENT_FILE']
	except KeyError:
		env_file = os.path.expanduser('~/.irods/irods_environment.json')

	ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH, cafile=None, capath=None, cadata=None)
	ssl_settings = {'ssl_context': ssl_context}
	pds = []
	with iRODSSession(irods_env_file=env_file, **ssl_settings) as session:
		try:
			coll = session.collections.get(irods_path)
		except CollectionDoesNotExist:
			logging.error(f"iRods path not found : {irods_path}")

		for quart_arch in coll.data_objects:
			all_file = b""
			with quart_arch.open('r') as f:
				with tarfile.open(mode='r:gz', fileobj=f) as extr_f:
					while True:
						print(extr_f.next())
				"""print(f)
				while True:
					chunk = f.read(1048576)
					all_file+= chunk
					if not chunk:
						break"""


#EROS_get_bad_timestamp('/eros/data/eros2/lightcurves/lm/lm001/lm0011','.')

# MACHO_gz_path = "/Volumes/DisqueSauvegarde/MACHO/lightcurves/F_42"
# for f in os.listdir(MACHO_gz_path):
# 	if f[-3:] == '.gz':
# 		print(f)
# 		MACHO_raw_to_pickle(f, MACHO_gz_path, "/Volumes/DisqueSauvegarde/working_dir/pickles/F_42")


df = pd.read_pickle("/Volumes/DisqueSauvegarde/working_dir/pickles/F_42/F_42.128.bz2")
print(df.time)
print(np.sort(np.load("/Volumes/DisqueSauvegarde/working_dir/pickles/F_42/42_red_M_ratios.npy")))

#MACHO_get_bad_timestamps(field=42, pickles_path="/Volumes/DisqueSauvegarde/working_dir/pickles", output_path="/Volumes/DisqueSauvegarde/working_dir/pickles/F_42")