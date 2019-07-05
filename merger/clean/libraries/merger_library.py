import pandas as pd
import numpy as np
import os
import gzip
import time
import logging
import tarfile
from irods.session import iRODSSession
from irods.exception import CollectionDoesNotExist, DataObjectDoesNotExist
import ssl

from requests import get

import dask.dataframe as dd
import dask.array as da


COLOR_FILTERS = {
	'red_E': {'mag': 'red_E', 'err': 'rederr_E'},
	'red_M': {'mag': 'red_M', 'err': 'rederr_M'},
	'blue_E': {'mag': 'blue_E', 'err': 'blueerr_E'},
	'blue_M': {'mag': 'blue_M', 'err': 'blueerr_M'}
}

OUTPUT_DIR_PATH = "/Volumes/DisqueSauvegarde/working_dir/"


def load_irods_eros_lightcurves(irods_filepath):
	"""
	Load EROS lightcurves from iRods storage.

	Load from individual .time files. I didn't check the time taken but I think it can be significantly longer than for other load methods.
	Parameters
	----------
	irods_filepath : str
		Path in the iRods directory containing the .time files (lm/lmXXX/lmXXXX/lmXXXXL)

	Returns
	-------
	pd.DataFrame
		Dataframe of the lightcurves (contains time, magnitudes and errors
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
			coll = session.collections.get(irods_filepath)
		except CollectionDoesNotExist:
			logging.error(f"iRods path not found : {irods_filepath}")
		for lcfile in coll.data_objects:
			id_E = lcfile.name
			if id_E[-4:] == 'time':
				with lcfile.open('r') as f:
					lc = {'time': [], 'red_E': [], 'rederr_E': [], 'blue_E': [], 'blueerr_E': [], 'id_E': []}
					for line in f.readlines()[4:]:
						line = line.decode().split()
						lc["time"].append(float(line[0]) + 49999.5)
						lc["red_E"].append(float(line[1]))
						lc["rederr_E"].append(float(line[2]))
						lc["blue_E"].append(float(line[3]))
						lc["blueerr_E"].append(float(line[4]))
						lc["id_E"].append(id_E[:-5])
				pds.append(pd.DataFrame.from_dict(lc))
		return pd.concat(pds)


def read_eros_lighcurve(filepath):
	"""

	Read one EROS lightcurve file and return it in a dataframe.

	Arguments:
		filepath {str} -- path to the EROS lightcurve (lm*.time)

	Returns:
		pandas DataFrame -- dataframe containing the EROS id, time of observation, magnitudes in blue and red and associated errors.
	"""
	try:
		with open(filepath) as f:
			for _ in range(4): f.readline()
			lc = {"time": [], "red_E": [], "rederr_E": [], "blue_E": [], "blueerr_E": [], "id_E": []}
			id_E = filepath.split('/')[-1][:-5]
			for line in f.readlines():
				line = line.split()
				lc["time"].append(float(line[0]) + 49999.5)
				lc["red_E"].append(float(line[1]))
				lc["rederr_E"].append(float(line[2]))
				lc["blue_E"].append(float(line[3]))
				lc["blueerr_E"].append(float(line[4]))
				lc["id_E"].append(id_E)
			f.close()
	except FileNotFoundError:
		logging.error(f"{filepath} doesn't exist.")
		return None
	return pd.DataFrame.from_dict(lc)


def load_eros_files(eros_path):
	"""[summary]

	[description]

	Arguments:
		eros_path {str} -- ideally path to an EROS CCD, but can be just a 1/4 or a whole field (not recommended as it uses a lot of memory).

	Returns:
		pandas DataFrame -- dataframe containing all the lightcurves in the subdirectories of eros_path
	"""
	pds = []
	for root, subdirs, files in os.walk(eros_path):
		print(subdirs)
		c = 0
		for filename in files:
			if filename[-4:] == "time":
				print(c, end='\r')
				c += 1
				# print(os.path.join(root, filename))
				pds.append(read_eros_lighcurve(os.path.join(root, filename)))
		print(c)
	return pd.concat(pds)


def read_compressed_eros_lightcurve(lc, exfile, name):
	"""Append data from EROS lightcurve to dict

	[description]

	Arguments:
		lc {dict} -- dictionnary containing the data of the current EROS 1/4 CCD
		exfile {file} -- lightcurve file
		name {str} -- lightcurve EROS identifier
	"""
	id_E = name.split("/")[-1][:-5]
	for line in exfile.readlines()[4:]:
		line = line.split()
		lc["time"].append(float(line[0]) + 49999.5)
		lc["red_E"].append(float(line[1]))
		lc["rederr_E"].append(float(line[2]))
		lc["blue_E"].append(float(line[3]))
		lc["blueerr_E"].append(float(line[4]))
		lc["id_E"].append(id_E)


def load_eros_compressed_files(filepath):
	"""Load EROS lightcurves from compressed tar.gz archives

	[description]

	Arguments:
		filepath {str} -- path to the file to open

	Returns:
		[DataFrame] -- dataframe containing all the lightcurves of "filepath"
	"""
	with tarfile.open(filepath, 'r:gz') as f:
		lc = {"time": [], "red_E": [], "rederr_E": [], "blue_E": [], "blueerr_E": [], "id_E": []}
		c = 0
		for member in f.getmembers():
			lcf = f.extractfile(member)
			if lcf:
				c += 1
				print(c, end='\r')
				read_compressed_eros_lightcurve(lc, lcf, member.name)
				lcf.close()
	return pd.DataFrame.from_dict(lc)


def read_macho_lightcurve(filepath, filename):
	"""
	Read MACHO lightcurves from tile archive.
	"""
	df = None
	lc = list()  # {'time':[], 'red_M':[], 'rederr_M':[], 'blue_M':[], 'blueerr_M':[], 'id_M':[]}
	try:
		with gzip.open(os.path.join(filepath, filename), 'rt') as f:
			linecount = 0
			for line in f:
				line = line.split(';')
				temp = list()
				temp.append(float(line[4]))
				temp.append(float(line[9]))
				temp.append(float(line[10]))
				temp.append(float(line[24]))
				temp.append(float(line[25]))
				temp.append(line[1] + ":" + line[2] + ":" + line[3])
				lc.append(tuple(temp))
				linecount += 1
				if linecount > 500000:
					linecount = 0
					lc = np.array(lc, dtype=[('time', 'f4'),
											 ('red_M', 'f4'),
											 ('rederr_M', 'f4'),
											 ('blue_M', 'f4'),
											 ('blueerr_M', 'f4'),
											 ('id_M', 'U13')])
					t = da.from_array(lc, chunks='100MB')
					lc = list()
					if df is None:
						df = t.to_dask_dataframe()
					else:
						df = dd.concat([df, t.to_dask_dataframe()])
			f.close()
	except FileNotFoundError:
		print(os.path.join(filepath, filename) + " doesn't exist.")
	if df is None:
		lc = np.array(lc, dtype=[('time', 'f4'),
								 ('red_M', 'f4'),
								 ('rederr_M', 'f4'),
								 ('blue_M', 'f4'),
								 ('blueerr_M', 'f4'),
								 ('id_M', 'U13')])
		t = da.from_array(lc, chunks='100MB')
		df = t.to_dask_dataframe()
	return df


def load_macho_from_url(filename):
	"""
	Load MACHO lightcurves from online database (http://macho.nci.org.au/macho_photometry)

	Parameters
	----------
	filename : str
		Name of file to load (F_ + field + . + tile + .gz). Example F_1.3319.gz

	Returns
	-------
	pd.DataFrame
	"""
	field = filename.split(".")[0]
	target_url = 'http://macho.nci.org.au/macho_photometry/' + field + '/' + filename
	try:
		file = gzip.decompress(get(target_url).content).decode().split('\n')[:-1]
	except OSError:
		logging.error(f"Not a gzipped file : {target_url}")
		return None

	# lc = {'time': [], 'red_M': [], 'rederr_M': [], 'blue_M': [], 'blueerr_M': [], 'id_M': []}
	lc = list()
	linecount = 0
	df = None
	for line in file:
		temp = list()
		line = line.split(';')
		temp.append(float(line[4]))
		temp.append(float(line[9]))
		temp.append(float(line[10]))
		temp.append(float(line[24]))
		temp.append(float(line[25]))
		temp.append(line[1] + ":" + line[2] + ":" + line[3])
		lc.append(tuple(temp))
		linecount += 1
		if linecount > 500000:
			linecount = 0
			lc = np.array(lc, dtype=[('time', 'f4'),
									 ('red_M', 'f4'),
									 ('rederr_M', 'f4'),
									 ('blue_M', 'f4'),
									 ('blueerr_M', 'f4'),
									 ('id_M', 'U13')])
			t = da.from_array(lc, chunks='100MB')
			lc = list()
			if df is None:
				df = t.to_dask_dataframe()
			else:
				df = dd.concat([df, t.to_dask_dataframe()])
	if df is None:
		lc = np.array(lc, dtype=[('time', 'f4'),
								 ('red_M', 'f4'),
								 ('rederr_M', 'f4'),
								 ('blue_M', 'f4'),
								 ('blueerr_M', 'f4'),
								 ('id_M', 'U13')])
		t = da.from_array(lc, chunks='100MB')
		df = t.to_dask_dataframe()
	return df


def load_macho_tiles(MACHO_files_path, field, tile_list):
	macho_path = MACHO_files_path + "F_" + str(field) + "/"
	pds = []
	for tile in tile_list:
		logging.debug(macho_path + "F_" + str(field) + "." + str(tile) + ".gz")
		# pds.append(pd.read_csv(macho_path+"F_49."+str(tile)+".gz", names=["id1", "id2", "id3", "time", "red_M", "rederr_M", "blue_M", "blueerr_M"], usecols=[1,2,3,4,9,10,24,25], sep=';'))
		if MACHO_files_path == 'url':
			pds.append(load_macho_from_url("F_" + str(field) + "." + str(tile) + ".gz"))
		else:
			pds.append(read_macho_lightcurve(macho_path, "F_" + str(field) + "." + str(tile) + ".gz"))
	if len(pds) == 1:
		return pds[0]
	return dd.concat(pds)


def load_macho_field(MACHO_files_path, field):
	macho_path = MACHO_files_path + "F_" + str(field) + "/"
	pds = []
	for root, subdirs, files in os.walk(macho_path):
		for file in files:
			if file[-2:] == 'gz':
				print(file)
				pds.append(read_macho_lightcurve(macho_path, file))
		# pds.append(pd.read_csv(os.path.join(macho_path+file), names=["id1", "id2", "id3", "time", "red_M", "rederr_M", "blue_M", "blueerr_M"], usecols=[1,2,3,4,9,10,24,25], sep=';'))
	return dd.concat(pds)


def merger_eros_first(output_dir_path, MACHO_field, eros_ccd, EROS_files_path, correspondance_files_path,
					  MACHO_files_path, quart="", save=True):
	"""
	Merge EROS and MACHO lightcurves, using EROS as starter

	Parameters
	----------
	output_dir_path : str
		Where to put resulting .pkl
	MACHO_field : int
		MACHO field on which merge lcs.
	eros_ccd : str
		ccd eros, format : "lm0***"

	Raises
	------
	NameError
		Did not find common stars in fields

	Returns
	-------
	pd.DataFrame
	"""
	start = time.time()

	# l o a d   E R O S
	logging.info("Loading EROS files")

	if EROS_files_path != 'irods':
		# eros_lcs = pd.concat([pd.read_pickle(output_dir_path+"full_"+eros_ccd+quart) for quart in 'klmn'])				# <===== Load from pickle files
		# eros_lcs = load_eros_files("/Volumes/DisqueSauvegarde/EROS/lightcurves/lm/"+eros_ccd[:5]+"/"+eros_ccd)			# <===== Load from .time files
		if quart not in "klmn" or quart == "":
			eros_lcs = pd.concat([load_eros_compressed_files(
				os.path.join(EROS_files_path, eros_ccd[:5], eros_ccd + quart + "-lc.tar.gz")) for quart in "klmn"])
		else:
			eros_lcs = load_eros_compressed_files(
				os.path.join(EROS_files_path, eros_ccd[:5], eros_ccd + quart + "-lc.tar.gz"))
	else:
		IRODS_ROOT = '/eros/data/eros2/lightcurves/lm/'
		if quart not in 'klmn' or quart == '':
			eros_lcs = pd.concat(
				[load_irods_eros_lightcurves(os.path.join(IRODS_ROOT, eros_ccd[:5], eros_ccd, eros_ccd + quart)) for
				 quart in "klmn"])
		else:
			eros_lcs = load_irods_eros_lightcurves(os.path.join(IRODS_ROOT, eros_ccd[:5], eros_ccd, eros_ccd + quart))
	end_load_eros = time.time()
	logging.info(str(end_load_eros - start) + ' seconds elapsed for loading EROS files')
	logging.info(f'{len(eros_lcs)} lines loaded.')

	# loading correspondance file and merging with load EROS stars
	logging.info("Merging")
	correspondance_path = os.path.join(correspondance_files_path, str(MACHO_field) + ".txt")
	correspondance = pd.read_csv(correspondance_path, names=["id_E", "id_M"], usecols=[0, 3], sep=' ')
	merged1 = eros_lcs.merge(correspondance, on="id_E", validate="m:1")
	del eros_lcs

	# determine needed tiles from MACHO
	tiles = np.unique([x.split(":")[1] for x in merged1.id_M.unique()])
	if not tiles.size:
		logging.error(f'No common stars in field, correspondace path : {correspondance_path}')
		raise NameError("No common stars in field !!!!")

	# l o a d   M A C H O
	logging.info("Loading MACHO files")
	macho_lcs = load_macho_tiles(MACHO_files_path, MACHO_field, tiles)

	logging.info("Merging")
	merged2 = macho_lcs.merge(correspondance, on='id_M', validate="m:1")
	del macho_lcs
	merged = pd.concat([merged1, merged2], copy=False)
	del merged1
	del merged2

	# replace invalid values in magnitudes with numpy NaN and remove rows with no valid magnitudes
	merged = merged.replace(to_replace=[99.999, -99.], value=np.nan).dropna(axis=0, how='all',
																			subset=['blue_E', 'red_E', 'blue_M',
																					'red_M'])

	# remove lightcurves missing one or more color
	merged = merged.groupby('id_E').filter(lambda x: x.red_E.count() != 0
													 and x.red_M.count() != 0
													 and x.blue_E.count() != 0
													 and x.blue_M.count() != 0)

	# save merged dataframe
	if save:
		logging.info("Saving")
		merged.to_pickle(os.path.join(output_dir_path, str(MACHO_field) + "_" + str(eros_ccd) + quart + ".pkl"))

	return merged


from dask.distributed import Client


def merger_macho_first(output_dir_path, MACHO_field, MACHO_tile, EROS_files_path, correspondance_files_path,
					   MACHO_files_path, save=True):
	client = Client(n_workers=1, threads_per_worker=1, processes=False, memory_limit='4GB')

	logging.info("Loading MACHO files")

	macho_lcs = load_macho_tiles(MACHO_files_path, MACHO_field, [MACHO_tile])

	# loading correspondance file and merging with load MACHO stars
	logging.info("Merging")
	correspondance_path = os.path.join(correspondance_files_path, str(MACHO_field) + ".txt")
	correspondance = dd.read_csv(correspondance_path, names=["id_E", "id_M"], usecols=[0, 3], sep=' ')
	merged1 = macho_lcs.merge(correspondance, on="id_M")
	del macho_lcs

	logging.info("Loading EROS lightcurves")
	if EROS_files_path == 'irods':
		st1 = time.time()
		IRODS_ROOT = '/eros/data/eros2/lightcurves/lm/'
		try:
			env_file = os.environ['IRODS_ENVIRONMENT_FILE']
		except KeyError:
			env_file = os.path.expanduser('~/.irods/irods_environment.json')

		ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH, cafile=None, capath=None,
												 cadata=None)
		ssl_settings = {'ssl_context': ssl_context}
		pds = []
		with iRODSSession(irods_env_file=env_file, **ssl_settings) as session:
			for id_E in merged1.id_E.unique().compute():
				logging.info(str(id_E))
				irods_filepath = os.path.join(IRODS_ROOT, id_E[:5], id_E[:6], id_E[:7], id_E + ".time")
				if irods_filepath[-4:] == 'time':
					st2 = time.time()
					try:
						obj = session.data_objects.get(irods_filepath)
					except DataObjectDoesNotExist:
						logging.error(f"iRods file not found : {irods_filepath}")
					with obj.open('r') as f:
						lc = {'time': [], 'red_E': [], 'rederr_E': [], 'blue_E': [], 'blueerr_E': [], 'id_E': []}
						for line in f.readlines()[4:]:
							line = line.decode().split()
							lc["time"].append(float(line[0]) + 49999.5)
							lc["red_E"].append(float(line[1]))
							lc["rederr_E"].append(float(line[2]))
							lc["blue_E"].append(float(line[3]))
							lc["blueerr_E"].append(float(line[4]))
							lc["id_E"].append(id_E)
					pds.append(pd.DataFrame.from_dict(lc))
					logging.info(time.time() - st2)
		eros_lcs = pd.concat(pds)
		del pds
		logging.info(f"{time.time() - st1} seconds to load {eros_lcs.id_E.nunique()}.")
	else:
		raise logging.error("Usual EROS loading not implemented yet !")
	eros_lcs = dd.from_pandas(eros_lcs, chunksize=500000)

	logging.info("Merging")
	merged2 = eros_lcs.merge(correspondance, on='id_E')
	del eros_lcs

	merged = dd.concat([merged1, merged2])

	# replace invalid values in magnitudes with numpy NaN and remove rows with no valid magnitudes
	merged = merged.replace(to_replace=[99.999, -99.], value=np.nan).dropna(how='all',
																			subset=['blue_E', 'red_E', 'blue_M',
																					'red_M'])

	# remove lightcurves missing one or more color
	rEs = merged.groupby('id_E')['red_E'].size()
	bEs = merged.groupby('id_E')['blue_E'].size()
	rMs = merged.groupby('id_E')['red_M'].size()
	bMs = merged.groupby('id_E')['blue_M'].size()
	merged = merged[merged['id_E'].map(rEs * bEs * rMs * bMs) > 0]
	# merged = merged.groupby('id_E').filter(lambda x: x.red_E.count() != 0
	# 												 and x.red_M.count() != 0
	# 												 and x.blue_E.count() != 0
	# 												 and x.blue_M.count() != 0)
	# save merged dataframe
	if save:
		logging.info("Saving")
		merged.to_pickle(os.path.join(output_dir_path, str(MACHO_field) + "_" + str(MACHO_tile) + ".pkl"))

	df = merged.compute()
	client.close()
	return df