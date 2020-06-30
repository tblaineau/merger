import numpy as np
import gzip
import os
import logging
import pandas as pd

from irods.session import iRODSSession
from irods.exception import CollectionDoesNotExist, DataObjectDoesNotExist
import ssl
import tarfile

logging.basicConfig(level=logging.INFO)

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


import io

def convert_eros_id(erosid):
	erosid = erosid.replace('lm', '1')
	for k, v in [('k', '0'), ('l', '1'), ('m', '2'), ('n', '3')]:
		erosid = erosid.replace(k, v)
	return erosid[:6]+erosid[6:].zfill(6)


def EROS_load_ccd(irods_path):
	"""
	Get lightcurves from one cdd of one EROS field

	Parameters
	----------
	irods_path : str
		Path to irods ccd directory containting the .tar.gz
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
		logging.info("Starting loading ...")
		all_lcs = []
		for quart_arch in coll.data_objects:
			with quart_arch.open('r') as f:
				all_file = b""
				#with tarfile.open(mode='r:gz', fileobj=f) as extr_f:
					#while True:
						#print(extr_f.next())
				while True:
					logging.debug("Chunk read")
					chunk = f.read(1048576)
					all_file+= chunk
					if not chunk:
						break
				all_file = io.BytesIO(all_file)
				with tarfile.open(mode='r:gz', fileobj=all_file) as extr_f:
					for file in extr_f.getmembers():
						f = extr_f.extractfile(file)
						if f:
							content = np.loadtxt(f, dtype=np.float64, comments="#")
							s = np.empty((content.shape[0], content.shape[1]+1))
							s[:,:-1] = content
							s[:,-1] = convert_eros_id(os.path.split(file.name)[1].split('.')[0])
							all_lcs.append(s)
			logging.info("New quart")
		logging.info("Loading completed.")
		all_lcs = np.concatenate(all_lcs)
		return pd.DataFrame(all_lcs, columns=["time", "red_E", "rederr_E", "blue_E", "blueerr_E", "id_E"])


def EROS_get_bad_timestamp(fccd, output_path=".", irods_path="/eros/data/eros2/lightcurves/lm/"):
	"""
	Return bad timestamps

	Parameters
	----------
	fccd : int or str
		Formated as FIELD+QUART
	irods_path : str
		path to the lightcurves
	output_path : str
		direcetory where to save timestamps
	"""
	if isinstance(fccd, str):
		pass
	elif isinstance(fccd, int):
		fccd = str(fccd).zfill(4)
	else:
		logging.error(f"Bad formatting for fccd : {fccd}")
	end_path = os.path.join("lm" + fccd[:-1], "lm" + fccd)
	df = EROS_load_ccd(os.path.join(irods_path, end_path))
	df.replace(to_replace=[99.999, 9.999], value=np.nan, inplace=True)

	df.dropna(axis='index', how='all', subset=['red_E', 'blue_E'], inplace=True)

	# Compute distance of each points. It is the difference between the median value of the lightcurve and the value of the point divided by the error of the point
	df[['median_red_E', 'median_blue_E']] = df.groupby('id_E')[['red_E', 'blue_E']].transform('median')
	# TODO : Compare to sigma intrinsic rather than vanilla errors !
	df['red_distance'] = (df['median_red_E'] - df['red_E']) / df['rederr_E']
	df['blue_distance'] = (df['median_blue_E'] - df['blue_E']) / df['blueerr_E']

	red_ratios = df.groupby('time')['red_distance'].agg(lambda x: x[x.abs() > 5].count() / x.count())
	blue_ratios = df.groupby('time')['blue_distance'].agg(lambda x: x[x.abs() > 5].count() / x.count())

	np.save(os.path.join(output_path, str(fccd) + "_red_E_ratios"), red_ratios)
	np.save(os.path.join(output_path, str(fccd) + "_blue_E_ratios"), blue_ratios)



# MACHO_gz_path = "/Volumes/DisqueSauvegarde/MACHO/lightcurves/F_42"
# for f in os.listdir(MACHO_gz_path):
# 	if f[-3:] == '.gz':
# 		print(f)
# 		MACHO_raw_to_pickle(f, MACHO_gz_path, "/Volumes/DisqueSauvegarde/working_dir/pickles/F_42")

#EROS_get_bad_timestamp(fccd=510, output_path=".")

#MACHO_get_bad_timestamps(field=42, pickles_path="/Volumes/DisqueSauvegarde/working_dir/pickles", output_path="/Volumes/DisqueSauvegarde/working_dir/pickles/F_42")