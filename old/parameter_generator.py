import numpy as np
import astropy.units as units
import astropy.constants as constants
import matplotlib.pyplot as plt
from scipy.integrate import nquad, dblquad, quad
from scipy.signal import find_peaks
import time

import numba as nb

MASS=1000.

r_lmc = 55000
r_earth = (150*1e6<<units.km).to(units.pc).value
r_0 = np.sqrt(4*constants.G/(constants.c**2)*(r_lmc<<units.pc)).decompose([units.Msun, units.pc]).value
pc_to_km = (units.pc.to(units.km))
t_obs = ((52697 - 48928) << units.d).to(units.s).value
a = 5000.  # pc
rho_0 = 0.0079  # M_sol/pc^3
d_sol = 8500  # pc
l_lmc, b_lmc = 280.4652 / 180. * np.pi, -32.8884 / 180. * np.pi
r_lmc = 55000  # pc
cosb_lmc = np.cos(b_lmc)
cosl_lmc = np.cos(l_lmc)

A = d_sol ** 2 + a ** 2
B = d_sol * cosb_lmc * cosl_lmc

@nb.jit
def rho_halo(x):
	return rho_0*A/((x*r_lmc)**2-2*x*r_lmc*B+A)

@nb.jit
def r(mass=MASS):
	R_0 = r_0*np.sqrt(mass)
	return r_earth/R_0

@nb.jit
def p_xvt(x, v_T, mass=MASS):
	u_lim = 1
	R_E = r_0*np.sqrt(mass*x*(1-x))
	return rho_halo(x)/mass*r_lmc*(2*u_lim*R_E*t_obs*np.abs(v_T))


@nb.jit
def f_vt(v_T, v0=220):
	return abs(2*v_T/(v0**2))*np.exp(-v_T**2/(v0**2))


@nb.jit
def x_from_delta_u(delta_u, mass=MASS):
	return r(mass)**2/(r(mass)**2+delta_u**2)


@nb.jit
def v_T_from_tEdu(delta_u, t_E, mass=MASS):
	R_0 = r_0*np.sqrt(mass)*pc_to_km
	ri = r(mass)
	return R_0/(t_E*86400) * ri*delta_u/(ri**2+delta_u**2)


@nb.jit
def jacobian(delta_u, t_E, mass=MASS):
	R_0 = r_0*np.sqrt(mass)*pc_to_km
	ri = r(mass)
	h1 = -2*ri**2*delta_u/(ri**2+delta_u**2)**2
	sqrth2 = ri*delta_u/(ri**2+delta_u**2)
	return -h1*sqrth2*R_0/(t_E*86400)**2


@nb.jit
def pdf_xvt(x, vt, mass=MASS):
	return p_xvt(x, vt, mass)*rho_halo(x)/mass*x*x*f_vt(vt)


@nb.jit
def pdf_tEdu(delta_u, t_E, mass=MASS):
	x = x_from_delta_u(delta_u, mass)
	vt = v_T_from_tEdu(delta_u, t_E, mass)
	return pdf_xvt(x, vt, mass)*np.abs(jacobian(delta_u, t_E, mass))

WORKING_DIR_PATH = "/Volumes/DisqueSauvegarde/working_dir/"

PERIOD_EARTH = 365.2422
alphaS = 80.8941667*np.pi/180.
deltaS = -69.7561111*np.pi/180.
epsilon = (90. - 66.56070833)*np.pi/180.
t_origin = 51442 #(21 septembre 1999) #58747 #(21 septembre 2019)

@nb.jit
def parallax(t, mag, u0, t0, tE, delta_u, theta):
	sin_beta = np.cos(epsilon)*np.sin(deltaS) - np.sin(epsilon)*np.cos(deltaS)*np.sin(alphaS)
	beta = np.arcsin(sin_beta) #ok because beta is in -pi/2; pi/2
	if abs(beta)==np.pi/2:
		lambda_star = 0
	else:
		lambda_star = np.sign((np.sin(epsilon)*np.sin(deltaS)+np.cos(epsilon)*np.sin(alphaS)*np.cos(deltaS))/np.cos(beta)) * np.arccos(np.cos(deltaS)*np.cos(alphaS)/np.cos(beta))
	tau = (t-t0)/tE
	phi = 2*np.pi * (t-t_origin)/PERIOD_EARTH - lambda_star
	u_D = np.array([
		-u0*np.sin(theta) + tau*np.cos(theta),
		 u0*np.cos(theta) + tau*np.sin(theta)
		])
	u_t = np.array([
		-delta_u*np.sin(phi),
		 delta_u*np.cos(phi)*sin_beta
		])
	u = np.linalg.norm(u_D-u_t, axis=0)
	return (u**2+2)/(u*np.sqrt(u**2+4))

@nb.jit
def microlens(t, params):
	mag, blend, u0, t0, tE, delta_u, theta = params
	return - 2.5*np.log10(blend*np.power(10, mag/-2.5) + (1-blend)*np.power(10, mag/-2.5) * parallax(t, mag, u0, t0, tE, delta_u, theta))

@nb.jit
def delta_u_from_x(x, mass):
	return r(mass) * np.sqrt((1 - x) / x)

kms_to_pcd = (units.km/units.s).to(units.pc/units.day)

@nb.jit
def tE_from_xvt(x, vT, mass):
	R_E = r_0*np.sqrt(mass*x*(1-x))
	return R_E/(vT*kms_to_pcd)

MASS=60.

time_range = np.linspace(48928, 52697, 10000)
t0=50000
mag=19

params = {
	'mag':mag,
	'blend':0.,
	'u0':0.1,
	't0':t0,
	'tE':100,
	'delta_u':0.5,	#no parallax
	'theta':20.*np.pi/180.
}

#@nb.jit(nopython=True)
def pdf_xvt_infdist(x, vt, u0, theta, mag=19., blend=0., t0=50000., mass=MASS, dist_threshold=2, time_range=np.linspace(48928, 52697, 10000), min_prominence=0):
	delta_u = delta_u_from_x(x, mass)
	tE = tE_from_xvt(x, vt, mass)
	c = microlens(time_range, [mag, blend, u0, t0, tE, delta_u, theta])
	peaks, _ = find_peaks(mag-c, prominence=min_prominence)
	if len(peaks)>dist_threshold:
		return pdf_xvt(x, vt)
	return 0

resolution = 1000
vT = np.linspace(-600, 600, resolution)
x = np.linspace(0, 1, resolution)
st1 = time.time()
#tot = dblquad(pdf_xvt, -np.inf, np.inf, gfun=lambda x: 0, hfun=lambda x: 1, epsrel=0.01)
#tot = dblquad(pdf_xvt_infdist, -np.inf, np.inf, gfun=lambda x: 0, hfun=lambda x: 1, args=(params,), epsrel=0.1)
tot = nquad(pdf_xvt_infdist, ranges=[(0,1), (-np.inf, np.inf), (0, 1), (0, 360*np.pi/180.)], opts={'epsrel':0.5})
st2 = time.time()
print(tot)
print(str(st2-st1)+" sec")
plt.imshow(pdf_xvt(x[None, :], vT[:, None], mass=MASS)/tot[0], extent=[x[0], x[-1], vT[0], vT[-1]], aspect='auto', origin='lower', cmap='inferno')
plt.xlabel(r'$x$')
plt.ylabel(r'$v_T [km/s]$')
plt.show()

#TODO: 0.009705106119492435