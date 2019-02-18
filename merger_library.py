import pandas as pd
import os
import gzip

COLOR_FILTERS = {
	'red_E':{'mag':'red_E', 'err': 'rederr_E'},
	'red_M':{'mag':'red_M', 'err': 'rederr_M'},
	'blue_E':{'mag':'blue_E', 'err': 'blueerr_E'},
	'blue_M':{'mag':'blue_M', 'err': 'blueerr_M'}
}

def read_eros_lighcurve(filepath):
	f = open(filepath)
	for _ in range(4): f.readline()
	lc = {"time":[], "red_E":[], "rederr_E":[], "blue_E":[], "blueerr_E":[], "id_E":[]}
	for line in f.readlines():
		line = line.split()
		lc["time"].append(float(line[0])+49999.5)
		lc["red_E"].append(float(line[1]))
		lc["rederr_E"].append(float(line[2]))
		lc["blue_E"].append(float(line[3]))
		lc["blueerr_E"].append(float(line[4]))
		lc["id_E"].append(filepath.split('/')[-1][:-5])
	f.close()
	return pd.DataFrame.from_dict(lc)

def read_macho_lightcurve(filepath):
	with gzip.open(filepath, 'rt') as f:
		lc = {'time':[], 'red_M':[], 'rederr_M':[], 'blue_M':[], 'blueerr_M':[], 'id_M':[]}
		for line in f:
			line = line.split(';')
			lc['time'].append(float(line[4]))
			lc['red_M'].append(float(line[9]))
			lc['rederr_M'].append(float(line[10]))
			lc['blue_M'].append(float(line[24]))
			lc['blueerr_M'].append(float(line[25]))
			lc['id_M'].append(line[1]+":"+line[2]+":"+line[3])
		f.close()
	return pd.DataFrame.from_dict(lc)

def open_eros_files(eros_path):
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

def load_macho_tiles(field, tile_list):
	macho_path = "/Volumes/DisqueSauvegarde/MACHO/lightcurves/F_"+str(field)+"/"
	pds = []
	for tile in tile_list:
		print(macho_path+"F_"+str(field)+"."+str(tile)+".gz")
		# pds.append(pd.read_csv(macho_path+"F_49."+str(tile)+".gz", names=["id1", "id2", "id3", "time", "red_M", "rederr_M", "blue_M", "blueerr_M"], usecols=[1,2,3,4,9,10,24,25], sep=';'))
		pds.append(read_macho_lightcurve(macho_path+"F_49."+str(tile)+".gz"))
	return pd.concat(pds)

def load_macho_field(field):
	macho_path = "/Volumes/DisqueSauvegarde/MACHO/lightcurves/F_"+str(field)+"/"
	pds = []
	for root, subdirs, files in os.walk(macho_path):
		for file in files:
			if file[-2:]=='gz':
				print(file)
				pds.append(read_macho_lightcurve(macho_path+file))
				#pds.append(pd.read_csv(os.path.join(macho_path+file), names=["id1", "id2", "id3", "time", "red_M", "rederr_M", "blue_M", "blueerr_M"], usecols=[1,2,3,4,9,10,24,25], sep=';'))
	return pd.concat(pds)

def generate_microlensing_parameters(seed):
	tmin = 48928
	tmax = 52697
	seed = int(seed.replace('lm0', '').replace('k', '0').replace('l', '1').replace('m', '2').replace('n', '3'))
	np.random.seed(seed)
	u0 = np.random.uniform(0,1)
	tE = np.exp(np.random.uniform(6.21, 9.21))
	t0 = np.random.uniform(tmin-tE/2., tmax+tE/2.)
	return u0, t0, tE