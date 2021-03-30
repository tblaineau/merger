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
from dask.distributed import Client

import pkg_resources

COLOR_FILTERS = {
	'red_E':{'mag':'red_E', 'err': 'rederr_E', "flux":"flux_red_E", "fluxerr":"fluxerr_red_E"},
	'red_M':{'mag':'red_M', 'err': 'rederr_M', "flux":"flux_red_M", "fluxerr":"fluxerr_red_M"},
	'blue_E':{'mag':'blue_E', 'err': 'blueerr_E', "flux":"flux_blue_E", "fluxerr":"fluxerr_blue_E"},
	'blue_M':{'mag':'blue_M', 'err': 'blueerr_M', "flux":"flux_blue_M", "fluxerr":"fluxerr_blue_M"}
}

STAR_COUNT_PATH = pkg_resources.resource_filename('merger', 'utils/MACHOstarcounts')
STARS_PER_JOBS = 5000
OUTPUT_DIR_PATH = "/Volumes/DisqueSauvegarde/working_dir/"


def load_irods_eros_lightcurves(irods_filepath="", idE_list=[]):
	"""
	Load EROS lightcurves from iRods storage.

	Load from individual .time files. I didn't check the time taken but I think it can be significantly longer than for other load methods.
	Parameters
	----------
	irods_filepath : str
		Path in the iRods directory containing the .time files (lm/lmXXX/lmXXXX/lmXXXXL) (if loading a full CCD quarter)
	idE_list : list(str)
		List of EROS identifiers (lmXXXXLY....). Load only those stars

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
		if irods_filepath != "":
			for lcfile in coll.data_objects:
				id_E = lcfile.name
				if id_E[-4:]=='time':
					with lcfile.open('r') as f:
						lc = {'time':[], 'red_E':[], 'rederr_E':[], 'blue_E':[], 'blueerr_E':[], 'id_E':[]}
						for line in f.readlines()[4:]:
							line = line.decode().split()
							lc["time"].append(float(line[0])+49999.5)
							lc["red_E"].append(float(line[1]))
							lc["rederr_E"].append(float(line[2]))
							lc["blue_E"].append(float(line[3]))
							lc["blueerr_E"].append(float(line[4]))
							lc["id_E"].append(id_E[:-5])
					pds.append(pd.DataFrame.from_dict(lc))
		elif len(idE_list) != 0:
			IRODS_ROOT = '/eros/data/eros2/lightcurves/lm/'
			times = []
			for id_E in idE_list:
				logging.info(str(id_E))
				irods_filepath= os.path.join(IRODS_ROOT, id_E[:5], id_E[:6], id_E[:7], id_E + ".time")
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
					times.append(time.time() - st2)
					if (len(times)>10 and np.median(times[-10:]) >= 3) or times[-1]>=10.:
						logging.warning(f"EROS loading take too much time. No time to waste.")
						logging.warning(f"Restarting EROS loading session...")
						return "RESTART"
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
			lc = {"time":[], "red_E":[], "rederr_E":[], "blue_E":[], "blueerr_E":[], "id_E":[]}
			id_E = filepath.split('/')[-1][:-5]
			for line in f.readlines():
				line = line.split()
				lc["time"].append(float(line[0])+49999.5)
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
		c=0
		for filename in files:
			if filename[-4:]=="time":
				print(c, end='\r')
				c+=1
				#print(os.path.join(root, filename))
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
		lc["time"].append(float(line[0])+49999.5)
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
		lc = {"time":[], "red_E":[], "rederr_E":[], "blue_E":[], "blueerr_E":[], "id_E":[]}
		c = 0
		for member in f.getmembers():
			lcf = f.extractfile(member)
			if lcf:
				c+=1
				print(c, end='\r')
				read_compressed_eros_lightcurve(lc, lcf, member.name)
				lcf.close()
	return pd.DataFrame.from_dict(lc)


def read_macho_lightcurve(filepath, filename, star_nb_start=0, star_nb_stop=-1):
	"""
	Read MACHO lightcurves from tile archive.

	Parameters
	----------
	filepath : str
	filename : str
	star_nb_start : int
		From which star to start saving value of file, default : 0 (from first line)
	star_nb_stop : int
		Star at which the program will stop reading the file, default : -1 (goes to the end of file)

	Returns
	-------
	pd.DataFrame
	"""

	lc = list() # {'time':[], 'red_M':[], 'rederr_M':[], 'blue_M':[], 'blueerr_M':[], 'id_M':[]}
	curr_star_nb = 0
	try:
		with gzip.open(os.path.join(filepath, filename), 'rt') as f:
			line = f.readline().split(';')
			last_star_id = line[3]
			f.seek(0)

			#Discard until star_nb_start
			while curr_star_nb<star_nb_start:
				line = f.readline().split(';')
				if last_star_id != line[3]:
					last_star_id = line[3]
					curr_star_nb+=1

			#Save until star_nb_stop
			for line in f:
				line = line.split(';')
				temp = list()
				if last_star_id != line[3]:
					last_star_id = line[3]
					curr_star_nb += 1
					if curr_star_nb == star_nb_stop+1:
						break
				temp.append(float(line[4]))
				temp.append(float(line[9]))
				temp.append(float(line[10]))
				temp.append(float(line[24]))
				temp.append(float(line[25]))
				temp.append(float(line[17]))
				temp.append(float(line[32]))
				temp.append(line[1] + ":" + line[2] + ":" + line[3])
				lc.append(tuple(temp))
			f.close()
	except FileNotFoundError:
		logging.error(os.path.join(filepath, filename) + " doesn't exist.")
	lc = np.array(lc, dtype=[('time', 'f8'),
							 ('red_M', 'f8'),
							 ('rederr_M', 'f8'),
							 ('blue_M', 'f8'),
							 ('blueerr_M', 'f8'),
							 ('red_amp', int),
							 ('blue_amp', int),
							 ('id_M', 'U13')])
	return pd.DataFrame.from_records(lc)


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
	target_url = 'http://macho.nci.org.au/macho_photometry/'+field+'/'+filename
	try:
		file = gzip.decompress(get(target_url).content).decode().split('\n')[:-1]
	except OSError:
		logging.error(f"Not a gzipped file : {target_url}")
		return None

	#lc = {'time': [], 'red_M': [], 'rederr_M': [], 'blue_M': [], 'blueerr_M': [], 'id_M': []}
	lc = list()
	linecount=0
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
		if linecount>500000:
			linecount=0
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
	macho_path = MACHO_files_path+"F_"+str(field)+"/"
	pds = []
	for tile in tile_list:
		logging.debug(macho_path+"F_"+str(field)+"."+str(tile)+".gz")
		# pds.append(pd.read_csv(macho_path+"F_49."+str(tile)+".gz", names=["id1", "id2", "id3", "time", "red_M", "rederr_M", "blue_M", "blueerr_M"], usecols=[1,2,3,4,9,10,24,25], sep=';'))
		if MACHO_files_path=='url':
			pds.append(load_macho_from_url("F_"+str(field)+"."+str(tile)+".gz"))
		else:
			pds.append(read_macho_lightcurve(macho_path, "F_"+str(field)+"."+str(tile)+".gz"))
	return dd.concat(pds)


def load_macho_field(MACHO_files_path, field):
	macho_path = MACHO_files_path+"F_"+str(field)+"/"
	pds = []
	for root, subdirs, files in os.walk(macho_path):
		for file in files:
			if file[-2:]=='gz':
				print(file)
				pds.append(read_macho_lightcurve(macho_path, file))
				#pds.append(pd.read_csv(os.path.join(macho_path+file), names=["id1", "id2", "id3", "time", "red_M", "rederr_M", "blue_M", "blueerr_M"], usecols=[1,2,3,4,9,10,24,25], sep=';'))
	return dd.concat(pds)


def load_macho_stars(MACHO_files_path, MACHO_field, t_indice):
	"""
	Load MACHO stars by group of STARS_PER_JOBS stars.

	Parameters
	----------
	MACHO_files_path: str
		Path to MACHO tile files
	MACHO_field: int
		Curretn MACHO field
	t_indice: int
		Current job array number

	Returns
	-------
	pd.DataFrame
	"""
	counts = np.loadtxt(os.path.join(STAR_COUNT_PATH, "strcnt_"+str(MACHO_field)+".txt"), dtype=[('tile', 'i4'), ('number_of_stars', 'i4')])
	counts = counts[counts['number_of_stars'] > 0]	#No empty files
	start = (t_indice - 1) * STARS_PER_JOBS
	end = t_indice * STARS_PER_JOBS - 1
	tot_starcounts = counts['number_of_stars'].cumsum()
	start_idx = np.searchsorted(tot_starcounts, start, side="right")
	end_idx = np.searchsorted(tot_starcounts, end, side="right")
	start_tile = counts['tile'][start_idx]
	if end_idx >= len(tot_starcounts):
		end_tile = counts['tile'][-1]
		n_end = tot_starcounts[-1] - counts['number_of_stars'][-1]
	else:
		end_tile = counts['tile'][end_idx]
		n_end = tot_starcounts[end_idx] - counts['number_of_stars'][end_idx]

	full_path = os.path.join(MACHO_files_path, 'F_'+str(MACHO_field))
	n_start = tot_starcounts[start_idx] - counts['number_of_stars'][start_idx] 	#First cumulated star number
	if start_idx==end_idx:
		pds = read_macho_lightcurve(full_path, 'F_'+str(MACHO_field)+'.'+str(start_tile)+'.gz', start-n_start, end-n_end)
	else:
		pds = list()
		pds.append(read_macho_lightcurve(full_path, 'F_'+str(MACHO_field)+'.'+str(start_tile)+'.gz', star_nb_start=start-n_start))
		pds.append(read_macho_lightcurve(full_path, 'F_'+str(MACHO_field)+'.'+str(end_tile)+'.gz', star_nb_stop=end-n_end))
		if start_idx+1 <= end_idx-1:
			for idx in range(start_idx+1, end_idx):
				pds.append(read_macho_lightcurve(full_path, 'F_' + str(MACHO_field) + '.' + str(counts['tile'][idx]) + '.gz'))
	return pd.concat(pds, copy=False, sort=False)



def merger_eros_first(output_dir_path, MACHO_field, eros_ccd, EROS_files_path, correspondance_files_path, MACHO_files_path, quart="", save=True):
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
		if quart not in "klmn" or quart=="":
			eros_lcs = pd.concat([load_eros_compressed_files(os.path.join(EROS_files_path,eros_ccd[:5],eros_ccd+quart+"-lc.tar.gz")) for quart in "klmn"])
		else:
			eros_lcs = load_eros_compressed_files(os.path.join(EROS_files_path,eros_ccd[:5],eros_ccd+quart+"-lc.tar.gz"))
	else:
		IRODS_ROOT = '/eros/data/eros2/lightcurves/lm/'
		if quart not in 'klmn' or quart=='':
			eros_lcs = pd.concat([load_irods_eros_lightcurves(os.path.join(IRODS_ROOT, eros_ccd[:5], eros_ccd, eros_ccd+quart)) for quart in "klmn"])
		else:
			eros_lcs = load_irods_eros_lightcurves(os.path.join(IRODS_ROOT, eros_ccd[:5], eros_ccd, eros_ccd+quart))
	end_load_eros = time.time()
	logging.info(str(end_load_eros-start)+' seconds elapsed for loading EROS files')
	logging.info(f'{len(eros_lcs)} lines loaded.')

	#loading correspondance file and merging with load EROS stars
	logging.info("Merging")
	correspondance_path=os.path.join(correspondance_files_path, str(MACHO_field)+".txt")
	correspondance = pd.read_csv(correspondance_path, names=["id_E", "id_M"], usecols=[0, 3], sep=' ')
	merged1 = eros_lcs.merge(correspondance, on="id_E", validate="m:1")
	del eros_lcs

	# determine needed tiles from MACHO
	tiles = np.unique([x.split(":")[1] for x in merged1.id_M.unique()])
	if not tiles.size:
		logging.error(f'No common stars in field, correspondace path : {correspondance_path}')
		raise NameError("No common stars in field !!!!")

	#l o a d   M A C H O
	logging.info("Loading MACHO files")
	macho_lcs = load_macho_tiles(MACHO_files_path, MACHO_field, tiles)

	logging.info("Merging")
	merged2 = macho_lcs.merge(correspondance, on='id_M', validate="m:1")
	del macho_lcs
	merged = pd.concat((merged1, merged2), copy=False)
	del merged1
	del merged2

	# replace invalid values in magnitudes with numpy NaN and remove rows with no valid magnitudes
	merged = merged.replace(to_replace=[99.999,-99.], value=np.nan).dropna(axis=0, how='all', subset=['blue_E', 'red_E', 'blue_M', 'red_M'])

	# remove lightcurves missing one or more color
	merged = merged.groupby('id_E').filter(lambda x: x.red_E.count()!=0
		and x.red_M.count()!=0
		and x.blue_E.count()!=0
		and x.blue_M.count()!=0)

	# save merged dataframe
	if save:
		logging.info("Saving")
		merged.to_pickle(os.path.join(output_dir_path, str(MACHO_field)+"_"+str(eros_ccd)+quart+".pkl"), compression='bz2')

	return merged

def merger_macho_first(output_dir_path, MACHO_field, EROS_files_path, correspondance_files_path, MACHO_files_path, save=True, t_indice=None, MACHO_tile=None,):
	logging.info("Loading MACHO files")

	if isinstance(MACHO_tile, list) or isinstance(MACHO_tile, np.ndarray):
		macho_lcs = load_macho_tiles(MACHO_files_path, MACHO_field, [MACHO_tile])
	elif isinstance(t_indice, int):
		macho_lcs = load_macho_stars(MACHO_files_path=MACHO_files_path, MACHO_field=MACHO_field, t_indice=t_indice)
	else:
		logging.error("No tile or t_indice defined or bad format")
		raise SystemExit(0)


	# loading correspondance file and merging with load MACHO stars
	logging.info("Merging")
	correspondance_path = os.path.join(correspondance_files_path, str(MACHO_field) + ".txt")
	correspondance = pd.read_csv(correspondance_path, names=["id_E", "id_M"], usecols=[0, 3], sep=' ')
	merged1 = macho_lcs.merge(correspondance, on="id_M", validate="m:1")
	del macho_lcs

	logging.info("Loading EROS lightcurves")

	if EROS_files_path == 'irods':
		st1 = time.time()
		eros_lcs = ""
		for i in range(10):
			eros_lcs = load_irods_eros_lightcurves(idE_list=merged1.id_E.unique())
			if isinstance(eros_lcs, pd.DataFrame):
				break
			if i==9:
				raise SystemExit(f"iRods takes too much times. Restart the job.")
		logging.info(f"{time.time()-st1} seconds to load {eros_lcs.id_E.nunique()}.")
	else:
		raise logging.error("Usual EROS loading not implemented yet !")

	logging.info("Merging")
	merged2 = eros_lcs.merge(correspondance, on='id_E', validate="m:1")
	del eros_lcs

	merged = pd.concat((merged1, merged2), copy=False, sort=False)

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
		if MACHO_tile:
			name = MACHO_tile
		else:
			name = t_indice
		merged.to_pickle(os.path.join(output_dir_path, str(MACHO_field) + "_" + str(name) + ".bz2"), compression='bz2')

	return merged