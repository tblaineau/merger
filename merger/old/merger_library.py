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

def rejection_sampling(func, range_x, nb=1, max_sampling=100000, pdf_max=None, args=[]):
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
		max_funcx = np.max(func(x, *args))
	else:
		max_funcx = pdf_max

	while len(v)<nb:
		x=np.random.uniform(min_x, max_x);
		y=np.random.uniform(0, max_funcx);
		if x!=0 and y<func(x, *args):
			v.append(x)
	return v

class Microlensing_generator():
	def __init__(self, mass=100, u_lim=1, v0=220, blending=False, max_blend=0.7):
		self.mass = mass
		self.u_lim = u_lim
		self.blending = blending

		self.tmin = 48928
		self.tmax = 52697
		self.tobs = self.tmax - self.tmin
		self.r_lmc = 55000
		self.a = 5000
		self.rho_0 = 0.0079
		self.d_sol = 8500
		self.cosb_lmc = np.cos(-32.8884/180.*np.pi)
		self.cosl_lmc = np.cos(280.4652/180.*np.pi)
		self.A = self.d_sol**2 + self.a**2
		self.B = self.d_sol * self.cosb_lmc * self.cosl_lmc
		self.v0 = v0
		self.max_blend=0.7
		self.r_earth = (150*1e6 *units.km).to(units.pc).value

		self.r_0 = np.sqrt(4 * constants.G / (constants.c**2) * self.r_lmc*units.pc).decompose([units.Msun, units.pc]).value

		self.kms_to_pcd = (units.km/units.s).to(units.pc/units.d)

	def R_E(self, x):
		return self.r_0 * np.sqrt(self.mass*x*(1-x))

	def p_x(self, x, v_T):
		rho_halo = self.rho_0 * self.A / ((x * self.r_lmc)**2 - 2 * x * self.r_lmc * self.B + self.A)

		R_Ex = self.R_E(x)
		return rho_halo / self.mass * self.r_lmc * (2 * self.u_lim * R_Ex * self.tobs * v_T)

	def vt_ppf(self, x):
		return np.sqrt(-np.log(1 - x) * self.v0**2)


	def generate_parameters(self, seed=None):
		if seed:
			seed = int(seed.replace('lm0', '').replace('k', '0').replace('l', '1').replace('m', '2').replace('n', '3'))
			np.random.seed(seed)

		u0 = np.random.uniform(0, self.u_lim)
		v_T = self.vt_ppf(np.random.uniform())
		x = rejection_sampling(self.p_x, (0,1), nb=1, args=[v_T])[0]
		R_Ex = self.R_E(x)
		tE = R_Ex / (v_T * self.kms_to_pcd)
		t0 = np.random.uniform(self.tmin - tE/2., self.tmax + tE/2.)

		blend_factors = {}
		for key in COLOR_FILTERS.keys():
			if self.blending:
				blend_factors[key]=np.random.uniform(0, self.max_blend)
			else:
				blend_factors[key]=0

		theta = np.random.uniform(0, 2*np.pi)
		delta_u = self.r_earth * (1 - x) / R_Ex
		return u0, t0, tE, blend_factors, delta_u, theta

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