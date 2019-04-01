import pandas as pd
import os
import gzip
import numpy as np
import astropy.units as units
import astropy.constants as constants

COLOR_FILTERS = {
	'red_E':{'mag':'red_E', 'err': 'rederr_E'},
	'red_M':{'mag':'red_M', 'err': 'rederr_M'},
	'blue_E':{'mag':'blue_E', 'err': 'blueerr_E'},
	'blue_M':{'mag':'blue_M', 'err': 'blueerr_M'}
}

WORKING_DIR_PATH = "/Volumes/DisqueSauvegarde/working_dir/"

def read_eros_lighcurve(filepath):
	with open(filepath) as f:
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

def load_eros_files(eros_path):
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

def rho_halo_pdf(x):
	"""pdf of dark halo density
	
	[description]
	
	Arguments:
		x {float} -- x = d_OD/d_OS
	
	Returns:
		{float} -- dark matter density at x
	"""
	a=5000.			#pc
	rho_0=0.0079	#M_sol/pc^3
	d_sol = 8500	#pc
	l_lmc, b_lmc = 280.4652/180.*np.pi, -32.8884/180.*np.pi
	r_lmc = 55000	#pc
	cosb_lmc = np.cos(b_lmc)
	cosl_lmc = np.cos(l_lmc)
	A = d_sol**2+a**2
	B = d_sol*cosb_lmc*cosl_lmc
	return rho_0*A/((x*r_lmc)**2-2*x*r_lmc*B+A)*x*x

def vt_ppf(x, v0=220):
	"""ppf of transverse speed pdf
	
	ppf of p(v_T):
	p(v_T) = (2*v_T/(v0**2))*np.exp(-v_T**2/(v0**2))
	
	Arguments:
		x -- quantile
	
	Keyword Arguments:
		v0 {km/s} -- speed parameter (default: {220})
	
	Returns:
		int {km/s} -- corresponding speed
	"""
	return np.sqrt(-np.log(1-x)*v0*v0)

def rejection_sampling(func, range_x, nb=1, max_sampling=100000, pdf_max=None):
	"""generic rejection sampling algorithm
	
	[description]
	
	Arguments:
		func {function} -- probability density function
		range_x {(float, float)} -- range of x
	
	Keyword Arguments:
		nb {number} -- size of returned sample  (default: {1})
		max_sampling {number} -- size of sample for estimating the max of the pdf (default: {100000})
		pdf_max {float} -- use this as pdf maximum, if None, estimate it (default: {None})
	
	Returns:
		list -- sampled from the input pdf
	"""
	v=[]
	min_x, max_x = range_x
	if not pdf_max:
		x = np.linspace(min_x, max_x, max_sampling)
		max_funcx = np.max(func(x))
	else:
		max_funcx = pdf_max

	while len(v)<nb:
		x=np.random.uniform(min_x, max_x);
		y=np.random.uniform(0, max_funcx);
		if x!=0 and y<func(x):
			v.append(x)
	return v

max_rh0x = np.max(rho_halo_pdf(np.linspace(0, 1, 100000)))		#maximum value estimate of dark halo density function

def generate_physical_ml_parameters(seed, mass, u0_range=(0,1), blending=False):
	tmin = 48928
	tmax = 52697
	r_lmc = 55000*units.pc
	r_earth = 150*1e6*units.km
	seed = int(seed.replace('lm0', '').replace('k', '0').replace('l', '1').replace('m', '2').replace('n', '3'))
	np.random.seed(seed)

	u0 = np.random.uniform(*u0_range)
	x = rejection_sampling(rho_halo_pdf, (0,1), nb=1, pdf_max=max_rh0x)[0]
	v_T = vt_ppf(np.random.uniform())
	R_E = np.sqrt(4*constants.G*mass*units.M_sun/(constants.c**2)*r_lmc*x*(1-x))
	tE = R_E/(v_T*units.km/units.s)
	tE = tE.to(units.day).value
	t0 = np.random.uniform(tmin-tE/2., tmax+tE/2.)

	blend_factors = {}
	for key in COLOR_FILTERS.keys():
		if blending:
			blend_factors[key]=np.random.uniform(0, 0.7)
		else:
			blend_factors[key]=0

	theta = np.random.uniform(0, 2*np.pi)
	delta_u = (r_earth*(1-x)/R_E).decompose().value
	return u0, t0, tE, blend_factors, delta_u, theta

print(generate_physical_ml_parameters('lm0103n190', 100))

def generate_microlensing_parameters(seed, blending=False, parallax=False):
	tmin = 48928
	tmax = 52697
	seed = int(seed.replace('lm0', '').replace('k', '0').replace('l', '1').replace('m', '2').replace('n', '3'))
	np.random.seed(seed)
	u0 = np.random.uniform(0,1)
	tE = np.exp(np.random.uniform(6.21, 9.21))
	t0 = np.random.uniform(tmin-tE/2., tmax+tE/2.)
	blend_factors = {}
	for key in COLOR_FILTERS.keys():
		if blending:
			blend_factors[key]=np.random.uniform(0, 0.7)
		else:
			blend_factors[key]=0
	theta = 0
	delta_u = 0
	if parallax:
		theta = np.random.uniform(0, 2*np.pi)
		delta_u = np.random.uniform(0,1)
	return u0, t0, tE, blend_factors, delta_u, theta