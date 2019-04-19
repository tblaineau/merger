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

@nb.njit
def parallax(t, mag, u0, t0, tE, delta_u, theta):
	out = np.zeros(t.shape)
	for i in range(len(t)):
		ti = t[i]
		tau = (ti-t0)/tE
		phi = 2*np.pi * (ti-t_origin)/PERIOD_EARTH - lambda_star
		t1 = u0**2 + tau**2
		t2 = delta_u**2 * (np.sin(phi)**2 + np.cos(phi)**2*sin_beta**2)
		t3 = -2*delta_u*u0 * (np.sin(phi)*np.sin(theta) + np.cos(phi)*np.cos(theta)*sin_beta)
		t4 = 2*tau*delta_u * (np.sin(phi)*np.cos(theta) - np.cos(phi)*np.sin(theta)*sin_beta)
		u = np.sqrt(t1+t2+t3+t4)
		out[i] = (u**2+2)/(u*np.sqrt(u**2+4))
	return out

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

sin_beta = np.cos(epsilon)*np.sin(deltaS) - np.sin(epsilon)*np.cos(deltaS)*np.sin(alphaS)
beta = np.arcsin(sin_beta) #ok because beta is in -pi/2; pi/2
if abs(beta)==np.pi/2:
	lambda_star = 0
else:
	lambda_star = np.sign((np.sin(epsilon)*np.sin(deltaS)+np.cos(epsilon)*np.sin(alphaS)*np.cos(deltaS))/np.cos(beta)) * np.arccos(np.cos(deltaS)*np.cos(alphaS)/np.cos(beta))

@nb.njit
def nb_microlens(t_range, mag, blend, u0, t0, tE, delta_u, theta):
	out = np.zeros(t_range.shape)
	for i in range(len(t_range)):
		t = t_range[i]
		tau = (t-t0)/tE
		phi = 2*np.pi * (t-t_origin)/PERIOD_EARTH - lambda_star
		t1 = u0**2 + tau**2
		t2 = delta_u**2 * (np.sin(phi)**2 + np.cos(phi)**2*sin_beta**2)
		t3 = -2*delta_u*u0 * (np.sin(phi)*np.sin(theta) + np.cos(phi)*np.cos(theta)*sin_beta)
		t4 = 2*tau*delta_u * (np.sin(phi)*np.cos(theta) - np.cos(phi)*np.sin(theta)*sin_beta)
		u = np.sqrt(t1+t2+t3+t4)
		parallax  = (u**2+2)/(u*np.sqrt(u**2+4))
		out[i] = - 2.5*np.log10(blend*np.power(10, mag/-2.5) + (1-blend)*np.power(10, mag/-2.5) * parallax)
	return out


def pdf_xvt_infdist(x, vt, u0, theta, time_range=np.linspace(48928, 52697, 1000), mag=19., blend=0., t0=50000., mass=MASS, dist_threshold=2, min_prominence=0):
	delta_u = delta_u_from_x(x, mass)
	tE = tE_from_xvt(x, vt, mass)
	c = nb_microlens(time_range, mag, blend, u0, t0, tE, delta_u, theta)
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
t_range = np.linspace(48928, 52697, 1000)
# tot = nquad(pdf_xvt_infdist, ranges=[(0,1), (-np.inf, np.inf), (0, 1), (0, 360*np.pi/180.)], args=[t_range], opts={'epsrel':0.5})
tot = nquad(pdf_xvt, ranges=[(0,1), (-np.inf, np.inf)], opts={'epsrel':0.5})[0]*360*np.pi/180.
st2 = time.time()
print(tot)
print(str(st2-st1)+" sec")
plt.imshow(pdf_xvt(x[None, :], vT[:, None], mass=MASS)/tot[0], extent=[x[0], x[-1], vT[0], vT[-1]], aspect='auto', origin='lower', cmap='inferno')
plt.xlabel(r'$x$')
plt.ylabel(r'$v_T [km/s]$')
plt.show()

#TODO: 0.009705106119492435

# 0.5035429397763278
# (0.5035429397763278, 0.10013537088425556)
# 58203.84471487999 sec
# for tot = nquad(pdf_xvt_infdist, ranges=[(0,1), (-np.inf, np.inf), (0, 1), (0, 360*np.pi/180.)], args=[t_range], opts={'epsrel':0.5})
#
# (5.101654634398505, 1.3790603843369689) -> 32.05464144115733
# 0.45479512214660645 sec
