import numpy as np
import astropy.units as units
import astropy.constants as constants
import numba as nb
import logging
from parallax_simulator_pipeline.metropolis_hastings import metropolis_hastings

COLOR_FILTERS = {
	'red_E':{'mag':'red_E', 'err': 'rederr_E'},
	'red_M':{'mag':'red_M', 'err': 'rederr_M'},
	'blue_E':{'mag':'blue_E', 'err': 'blueerr_E'},
	'blue_M':{'mag':'blue_M', 'err': 'blueerr_M'}
}

a=5000
rho_0=0.0079
d_sol = 8500
l_lmc, b_lmc = 280.4652/180.*np.pi, -32.8884/180.*np.pi
r_lmc = 55000
r_earth = (150*1e6*units.km).to(units.pc).value
t_obs = ((52697 - 48928) << units.d).to(units.s).value

pc_to_km = (units.pc.to(units.km))
kms_to_pcd = (units.km/units.s).to(units.pc/units.d)

cosb_lmc = np.cos(b_lmc)
cosl_lmc = np.cos(l_lmc)
A = d_sol ** 2 + a ** 2
B = d_sol * cosb_lmc * cosl_lmc
r_0 = np.sqrt(4*constants.G/(constants.c**2)*r_lmc*units.pc).decompose([units.Msun, units.pc]).value

@nb.njit
def r(mass):
	R_0 = r_0*np.sqrt(mass)
	return r_earth/R_0

@nb.njit
def R_E(x, mass):
	return r_0*np.sqrt(mass*x*(1-x))

@nb.njit
def rho_halo(x):
	return rho_0*A/((x*r_lmc)**2-2*x*r_lmc*B+A)

@nb.njit
def f_vt(v_T, v0=220):
	return (2*v_T/(v0**2))*np.exp(-v_T**2/(v0**2))

@nb.njit
def p_xvt(x, v_T, mass):
	return rho_halo(x)/mass*r_lmc*(2*r_0*np.sqrt(mass*x*(1-x))*t_obs*v_T)

@nb.njit
def delta_u_from_x(x, mass):
	return r(mass)*np.sqrt((1-x)/x)

@nb.njit
def tE_from_xvt(x, vt, mass):
	return r_0 * np.sqrt(mass*x*(1-x)) / (vt*kms_to_pcd)

@nb.njit
def pdf_xvt(x, mass):
	if x[0]<0 or x[0]>1 or x[1]<0:
		return 0
	return p_xvt(x[0], x[1], mass)*f_vt(x[1])

@nb.njit
def randomizer_gauss(x):
	return np.array([np.random.normal(loc=x[0], scale=0.1), np.random.normal(loc=x[1], scale=300)])


class MicrolensingGenerator:
	"""
	Class to generate microlensing paramters

	Parameters
	----------
	xvt_file : str or int
		If a str : Path to file containing x - v_T pairs generated through the Hasting-Metropolis algorithm
		If int, number of xvt pairs to pool
	seed : int
		Seed used for numpy.seed
	tmin : (float, float) or np.array
		lower and upper limit t_0 or array indiviudal lower and upper limits
	max_blend : float
		maximum authorized blend, if max_blend=0, no blending
	"""
	def __init__(self, xvt_file=None, seed=None, trange = np.array([48928., 52697.]), u_max=2.,  max_blend=0., min_blend=0.):
		self.seed = seed
		self.xvt_file = xvt_file
		self.trange = np.array(trange)
		self.u_max = u_max
		self.max_blend = max_blend
		self.min_blend = min_blend
		self.blending = bool(max_blend)
		self.blend_pdf = None
		self.generate_mass = False

		if self.seed:
			np.random.seed(self.seed)

		if self.xvt_file:
			if isinstance(self.xvt_file, str):
				try:
					self.xvts = np.load(self.xvt_file)
				except FileNotFoundError:
					logging.error(f"xvt file not found : {self.xvt_file}")
			elif isinstance(self.xvt_file, int):
				logging.info(f"Generating {self.xvt_file} x-vt pairs... ")
				self.xvts = np.array(metropolis_hastings(pdf_xvt, randomizer_gauss, self.xvt_file, np.array([0.5, 100]), (10.)))
			else:
				logging.error(f"xvts can't be loaded or generated, check variable : {self.xvt_file}")

	def generate_parameters(self, mass=30., seed=None, nb_parameters=1):
		"""
		Generate a set of microlensing parameters, including parallax and blending using S-model and fixed mass

		Parameters
		----------
		seed : str
			Seed used for parameter generation (EROS id)
		mass : float
			mass for which generate paramters (\implies \delta_u, t_E)
		nb_parameters : int
			number of parameters set to generate, overridden by trange dimension
		Returns
		-------
		dict
			Dictionnary of lists containing the parameters set
		"""
		if seed:
			seed = int(seed.replace('lm0', '').replace('k', '0').replace('l', '1').replace('m', '2').replace('n', '3'))
			np.random.seed(seed)

		if len(self.trange.shape)>1:
			nb_parameters = self.trange.shape[0]

		if self.generate_mass:
			mass = np.random.uniform(0, 200, size=nb_parameters)
		else:
			mass = np.array([mass]*nb_parameters)

		u0 = np.random.uniform(0, self.u_max, size=nb_parameters)
		x, vt = self.xvts[np.random.randint(0, self.xvts.shape[0], size=nb_parameters)].T
		vt *= np.random.choice([-1., 1.], size=nb_parameters, replace=True)
		delta_u = delta_u_from_x(x, mass=mass)
		tE = tE_from_xvt(x, vt, mass=mass)
		t0 = np.random.uniform(self.trange[:,0], self.trange[:,1])
		theta = np.random.uniform(0, 2 * np.pi, size=nb_parameters)
		params = {
			'u0': u0,
			't0': t0,
			'tE': tE,
			'delta_u': delta_u,
			'theta': theta,
			'mass': mass,
			'x': x,
			'vt': vt,
		}

		blend = np.random.uniform(self.min_blend, self.max_blend, size=nb_parameters)
		for key in COLOR_FILTERS.keys():
			if self.blending:
				params['blend_'+key] = blend
			else:
				params['blend_'+key] = [0] * nb_parameters
		return params


#parallax parameters.
PERIOD_EARTH = 365.25
alphaS = 80.8941667*np.pi/180.
deltaS = -69.7561111*np.pi/180.
epsilon = (90. - 66.56070833)*np.pi/180.		#source in LMC
t_origin = 51442 #(21 septembre 1999) #58747 #(21 septembre 2019)

sin_beta = np.cos(epsilon)*np.sin(deltaS) - np.sin(epsilon)*np.cos(deltaS)*np.sin(alphaS)
beta = np.arcsin(sin_beta) #ok because beta is in -pi/2; pi/2
if abs(beta)==np.pi/2:
	lambda_star = 0
else:
	lambda_star = np.sign((np.sin(epsilon)*np.sin(deltaS)+np.cos(epsilon)*np.sin(alphaS)*np.cos(deltaS))/np.cos(beta)) * np.arccos(np.cos(deltaS)*np.cos(alphaS)/np.cos(beta))


@nb.njit
def microlens_parallax(t, mag, blend, u0, t0, tE, delta_u, theta):
	tau = (t-t0)/tE
	phi = 2*np.pi * (t-t_origin)/PERIOD_EARTH - lambda_star
	t1 = u0**2 + tau**2
	t2 = delta_u**2 * (np.sin(phi)**2 + np.cos(phi)**2*sin_beta**2)
	t3 = -2*delta_u*u0 * (np.sin(phi)*np.sin(theta) + np.cos(phi)*np.cos(theta)*sin_beta)
	t4 = 2*tau*delta_u * (np.sin(phi)*np.cos(theta) - np.cos(phi)*np.sin(theta)*sin_beta)
	u = np.sqrt(t1+t2+t3+t4)
	parallax  = (u**2+2)/(u*np.sqrt(u**2+4))
	return - 2.5*np.log10(blend*np.power(10, mag/-2.5) + (1-blend)*np.power(10, mag/-2.5) * parallax)


@nb.njit
def microlens_simple(t, mag, blend, u0, t0, tE, delta_u, theta):
	u = np.sqrt(u0*u0 + ((t-t0)**2)/tE/tE)
	amp = (u**2+2)/(u*np.sqrt(u**2+4))
	return - 2.5*np.log10(blend*np.power(10, mag/-2.5) + (1-blend)*np.power(10, mag/-2.5) * amp)
