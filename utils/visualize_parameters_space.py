import pandas as pd
import numpy as np
import astropy.units as units
import astropy.constants as constants

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time

from scipy.stats import rv_continuous
from scipy.integrate import dblquad
from scipy.misc import derivative

from mpl_toolkits.mplot3d import Axes3D

COLOR_FILTERS = {
	'red_E':{'mag':'red_E', 'err': 'rederr_E'},
	'red_M':{'mag':'red_M', 'err': 'rederr_M'},
	'blue_E':{'mag':'blue_E', 'err': 'blueerr_E'},
	'blue_M':{'mag':'blue_M', 'err': 'blueerr_M'}
}

WORKING_DIR_PATH = "/Volumes/DisqueSauvegarde/working_dir/"

def rho_halo_cdf(x):
	a=5000.			#pc
	rho_0=0.0079	#M_sol/pc^3
	d_sol = 8500	#pc
	l_lmc, b_lmc = 280.4652/180.*np.pi, -32.8884/180.*np.pi
	r_lmc = 55000	#pc
	cosb_lmc = np.cos(b_lmc)
	cosl_lmc = np.cos(l_lmc)

	A = d_sol**2+a**2
	B = d_sol*cosb_lmc*cosl_lmc
	delta = np.sqrt(A-B*B)

	def pt1(x):
		return - (A-2*B**2)*np.arctan((r_lmc*x-B)/delta)/delta
	def pt2(x):
		return B*np.log(r_lmc*x*(r_lmc*x-2*B)+A)+r_lmc*x

	return A/r_lmc**3*((pt1(x)+pt2(x))-(pt1(0)+pt2(0)))

norm_rho = rho_halo_cdf(1)

def rho_halo(x):
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

	return rho_0*A/((x*r_lmc)**2-2*x*r_lmc*B+A)

def p_x(x):
	return rho_halo(x)*np.sqrt(x*(1-x))

def p_vt(v_T, v0=220):
	#Proba to find v_T i
	return (2*v_T/(v0**2))*np.exp(-v_T**2/(v0**2))

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

max_rh0x = np.max(rho_halo(np.linspace(0, 1, 100000)))		#maximum value estimate of dark halo density function

def generate_physical_ml_parameters(seed=None, mass=100, u0_range=(0,1), blending=False):
	tmin = 48928
	tmax = 52697
	r_lmc = 55000*units.pc
	r_earth = 150*1e6*units.km
	if seed:
		seed = int(seed.replace('lm0', '').replace('k', '0').replace('l', '1').replace('m', '2').replace('n', '3'))
		np.random.seed(seed)

	u0 = np.random.uniform(*u0_range)
	x = rejection_sampling(rho_halo, (0,1), nb=1, pdf_max=max_rh0x)[0]
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

r_lmc = 55000
r_0 = np.sqrt(4*constants.G/(constants.c**2)*r_lmc*units.pc).decompose([units.Msun, units.pc]).value
kms_to_pcd = (units.km/units.s).to(units.pc/units.day)

def compute_tE(x, v_T):
	R_E = r_0*np.sqrt(MASS*x*(1-x))
	tE = R_E/(v_T*kms_to_pcd)
	return tE

# NB_PARAMS_SETS = 10000
# MASS = 100

# s1 = time.time()
# all_params_sets = np.array([generate_physical_ml_parameters(mass=MASS) for i in range(NB_PARAMS_SETS)])
# s2 = time.time()

# print(s2-s1)

# aps = all_params_sets
# print(aps)

# plt.scatter(aps[:,2], aps[:,4], marker='o', s=1)
# plt.show()

V_T_SAMPLING=1000
RHO_SAMPLING=1000
x_range =np.linspace(0.01,0.99,RHO_SAMPLING)
px = rho_halo(x_range)*np.sqrt(x_range*(1-x_range))
vt_range = np.linspace(0.01,1000,V_T_SAMPLING)
pv_T = p_vt(vt_range)
prob_field = px[None,:]*pv_T[:,None]
plt.subplot(221)
plt.plot(x_range, px)
plt.subplot(223)
plt.contourf(x_range, vt_range, prob_field)
plt.xlabel("x")
plt.ylabel("v_T")
plt.subplot(224)
plt.plot(pv_T, vt_range)
plt.show()

plt.imshow(prob_field, origin='lower', extent=[x_range[0], x_range[-1], vt_range[0], vt_range[1]])
plt.show()

MASS=100
r_earth = 150*1e6*units.km
r_lmc = 55000*units.pc
R_0 = np.sqrt(4*constants.G*MASS*units.M_sun/(constants.c**2)*r_lmc)
R_0_km = R_0.to(units.km).value
r = (r_earth/R_0).decompose().value


def p_delta_u(delta_u):
	def g_inverse(delta_u):
		return (r**2)/(r**2+delta_u**2)

	def g_prime(x):
		return -r/(2*x**2*np.sqrt((1-x)/x))

	x = g_inverse(delta_u)
	return np.where(delta_u==0, 0, p_x(x)/np.abs(g_prime(x)))

def p_xvT(v_T, x):
	return p(x) * p_vt(v_T)

class tE_generator(rv_continuous):
	def _cdf(self, tE):
		def lower_bound(x):
			return (R_0_km*np.sqrt(x*(1-x))/(tE*86400))
		return dblquad(p_xvT, 0, 1, lower_bound, 10000)[0]

tEgen = tE_generator()
plt.imshow(compute_tE(x_range[None,:], vt_range[:,None]), vmax=3000, norm=LogNorm())
plt.show()

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(x_range[None,:], vt_range[:,None], prob_field, cmap=plt.get_cmap('Reds'))
# plt.show()

delta_u = np.linspace(0,0.02,100)
tE = np.linspace(0, 4000, 100)
ptE = np.array([tEgen.pdf(x) for x in tE])
np.save('probte'+str(MASS)+'m_corr.npy', ptE)
ptE = np.load('probte'+str(MASS)+'m_corr.npy')
prob_field2 = p_delta_u(delta_u)[None,:]*ptE[:,None]
plt.subplot(221)
plt.plot(delta_u, p_delta_u(delta_u))
plt.subplot(224)
plt.plot(ptE, tE)
plt.subplot(223)
plt.contourf(delta_u, tE, prob_field2)
plt.xlabel(r"$\delta_u$")
plt.ylabel(r"$t_E$")
plt.show()

plt.imshow(prob_field2, origin='lower', extent=[delta_u[0], delta_u[-1], tE[0], tE[-1]], aspect='auto')
plt.show()