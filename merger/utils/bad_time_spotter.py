import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gzip, os
import time
import seaborn as sns
import wget
import tarfile
from astropy.io import fits

def read_macho_lightcurve(filepath, fraction):
	try:
		with gzip.open(filepath, 'rt') as f:
			lc = {'time':[], 
				  'red_M':[], 
				  'rederr_M':[], 
				  'blue_M':[], 
				  'blueerr_M':[], 
				  'id_M':[], 
				  'red_crowd':[], 
				  'blue_crowd':[], 
				  'red_avesky':[],
				  'blue_avesky':[], 
				  'airmass':[],
				  'red_fwhm':[],
				  'blue_fwhm':[],
				  'red_normsky':[],
				  'blue_normsky':[],
				  'red_chi2':[],
				  'blue_chi2':[],
				  'red_type':[],
				  'blue_type':[],
				  "observation_id":[]
				 }
			for line in f:
				line = line.split(';')
				lc['id_M'].append(line[1]+":"+line[2]+":"+line[3])
				lc['time'].append(float(line[4]))
				lc['red_M'].append(float(line[9]))
				lc['rederr_M'].append(float(line[10]))
				lc['blue_M'].append(float(line[24]))
				lc['blueerr_M'].append(float(line[25]))
				lc['red_crowd'].append(float(line[13]))
				lc['blue_crowd'].append(float(line[28]))
				lc['red_avesky'].append(float(line[20]))
				lc['blue_avesky'].append(float(line[35]))
				lc['airmass'].append(float(line[8]))
				lc['red_fwhm'].append(float(line[21]))
				lc['blue_fwhm'].append(float(line[36]))
				lc['red_normsky'].append(float(line[11]))
				lc['blue_normsky'].append(float(line[26]))
				lc['red_chi2'].append(float(line[14]))
				lc['blue_chi2'].append(float(line[29]))
				lc['red_type'].append(float(line[12]))
				lc['blue_type'].append(float(line[27]))
				lc['observation_id'].append(int(line[5]))
			f.close()
	except FileNotFoundError:
		print(filepath+" doesn't exist.")
		return None
	return pd.DataFrame.from_dict(lc)

def load_macho_tiles(MACHO_files_path, field, tile_list, fraction=0.1):
	macho_path = MACHO_files_path+"F_"+str(field)+"/"
	pds = []
	for tile in tile_list:
		print(macho_path+"F_"+str(field)+"."+str(tile)+".gz")
		# pds.append(pd.read_csv(macho_path+"F_49."+str(tile)+".gz", names=["id1", "id2", "id3", "time", "red_M", "rederr_M", "blue_M", "blueerr_M"], usecols=[1,2,3,4,9,10,24,25], sep=';'))
		pds.append(read_macho_lightcurve(macho_path+"F_"+str(field)+"."+str(tile)+".gz", fraction))
	return pd.concat(pds)

def load_macho_field(MACHO_files_path, field, fraction=0.1):
	macho_path = MACHO_files_path+"F_"+str(field)+"/"
	pds = []
	for root, subdirs, files in os.walk(macho_path):
		for file in files:
			if file[-2:]=='gz':
				print(file)
				pds.append(read_macho_lightcurve(macho_path+file, fraction))
				#pds.append(pd.read_csv(os.path.join(macho_path+file), names=["id1", "id2", "id3", "time", "red_M", "rederr_M", "blue_M", "blueerr_M"], usecols=[1,2,3,4,9,10,24,25], sep=';'))
	return pd.concat(pds)

MACHO_files_path = "/Volumes/DisqueSauvegarde/MACHO/lightcurves/"
# test
# field = 50
# tile_list = [9033, 9034, 9035, 9036]

field = 60
tile_list = [6859, 6988, 6989]

t = load_macho_tiles(MACHO_files_path, field, tile_list, 0.5)
print(t)

start=time.time()
t1 = t.replace(
	to_replace={'red_M':-99.,
				'blue_M':-99.,
				'red_chi2':999.,
				'blue_chi2':999.,
				'red_normsky':999.,
				'blue_normsky':999.}
	, value=np.nan)

t1['blueerr_M'] = np.where(t1.blueerr_M.between(0,9.999, inclusive=False), t1.blueerr_M, np.nan)
t1['rederr_M'] = np.where(t1.rederr_M.between(0,9.999, inclusive=False), t1.rederr_M, np.nan)

t1[['median_red_M', 'median_blue_M']]= t1.groupby('id_M')[['red_M', 'blue_M']].transform('median')

t1['reddist1'] = (t1['median_red_M']-t1['red_M'])/t1['rederr_M']
t1['bluedist1'] = (t1['median_blue_M']-t1['blue_M'])/t1['blueerr_M']
print(time.time()-start)

start=time.time()
t1[['rederr_M_std', 'blueerr_M_std']] = t1.groupby('id_M')[['rederr_M', 'blueerr_M']].transform('std')
t1['reddist2'] = t1['rederr_M_std']/t1['rederr_M']
t1['bluedist2'] = t1['blueerr_M_std']/t1['blueerr_M']
print(time.time()-start)

stats1 = t1.groupby('id_M')[['red_M', 'blue_M', 'rederr_M', 'blueerr_M']].agg(['mean', 'std', 'median'])

ratio_dist = t1.groupby('time')['reddist1'].agg(lambda x: x[x>5].count()/x.count())
output = ratio_dist[ratio_dist >= ratio_dist.dropna().quantile(0.95)]
print(output)

output = ratio_dist[ratio_dist >= ratio_dist.dropna().quantile(0.95)]
print(output)
plt.rcParams['figure.figsize'] = [10, 10]
for curr_time in output.index:
	print(curr_time)
	oid = int(t1[t1.time==curr_time].observation_id.iloc[0])
	url = 'http://macho.nci.org.au/macho_images/O_'
	file = 'Obs_'+str(oid)+'-WCS-MEF.tar.gz'
	print(file)
	urlpath = url+str(oid//1000)+'/Obs_'+str(oid)+'-WCS-MEF.tar.gz'
	if not os.path.isfile(file):
		file = wget.download(urlpath)
	f = tarfile.open(file, 'r:gz')
	blue_f, red_f = f.getmembers()
	hdul = fits.open(f.extractfile(blue_f))
	plt.figure()

	plt.subplot(244)
	data = hdul[1].data
	low = np.percentile(np.array(data).flatten(), 15)
	high = np.percentile(np.array(data).flatten(), 99)
	plt.imshow(data, vmin=low, vmax=high)
	plt.subplot(243)
	data = hdul[2].data
	low = np.percentile(np.array(data).flatten(), 15)
	high = np.percentile(np.array(data).flatten(), 99)
	plt.imshow(data, vmin=low, vmax=high)
	plt.subplot(242)
	data = hdul[3].data
	low = np.percentile(np.array(data).flatten(), 15)
	high = np.percentile(np.array(data).flatten(), 99)
	plt.imshow(data, vmin=low, vmax=high)
	plt.subplot(241)
	data = hdul[4].data
	low = np.percentile(np.array(data).flatten(), 15)
	high = np.percentile(np.array(data).flatten(), 99)
	plt.imshow(data, vmin=low, vmax=high)
	plt.subplot(245)
	data = hdul[5].data
	low = np.percentile(np.array(data).flatten(), 15)
	high = np.percentile(np.array(data).flatten(), 99)
	plt.imshow(data, vmin=low, vmax=high)
	plt.subplot(246)
	data = hdul[6].data
	low = np.percentile(np.array(data).flatten(), 15)
	high = np.percentile(np.array(data).flatten(), 99)
	plt.imshow(data, vmin=low, vmax=high)
	plt.subplot(428)
	data = hdul[7].data.T
	low = np.percentile(np.array(data).flatten(), 15)
	high = np.percentile(np.array(data).flatten(), 99)
	plt.imshow(data, vmin=low, vmax=high)
	plt.subplot(426)
	data = hdul[8].data.T
	low = np.percentile(np.array(data).flatten(), 15)
	high = np.percentile(np.array(data).flatten(), 99)
	plt.imshow(data, vmin=low, vmax=high)
	plt.show()