import numpy as np
import astropy.units as units
import astropy.constants as constants

import matplotlib.pyplot as plt

MASS=1000

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

def rho_halo(x):
	return rho_0*A/((x*r_lmc)**2-2*x*r_lmc*B+A)

def r(mass=MASS):
	R_0 = r_0*np.sqrt(mass)
	return r_earth/R_0

def p_xvt(x, v_T, mass=MASS):
	u_lim = 1
	R_E = r_0*np.sqrt(mass*x*(1-x))
	return rho_halo(x)/mass*r_lmc*(2*u_lim*R_E*t_obs*np.abs(v_T))

def f_vt(v_T, v0=220):
	return (2*v_T/(v0**2))*np.exp(-v_T**2/(v0**2))

def x_from_delta_u(delta_u, mass=MASS):
	return r(mass)**2/(r(mass)**2+delta_u**2)

def v_T_from_tEdu(delta_u, t_E, mass=MASS):
	R_0 = r_0*np.sqrt(mass)*pc_to_km
	ri = r(mass)
	return R_0/(t_E*86400) * ri*delta_u/(ri**2+delta_u**2)

def jacobian(delta_u, t_E, mass=MASS):
	R_0 = r_0*np.sqrt(mass)*pc_to_km
	ri = r(mass)
	h1 = -2*ri**2*delta_u/(ri**2+delta_u**2)**2
	sqrth2 = ri*delta_u/(ri**2+delta_u**2)
	return -h1*sqrth2*R_0/(t_E*86400)**2

def pdf_xvt(x, vt, mass=MASS):
	return p_xvt(x, vt, mass)*rho_halo(x)/mass*x*x*f_vt(vt)

def p_tEdu(delta_u, t_E, mass=MASS):
	x = x_from_delta_u(delta_u, mass)
	vt = v_T_from_tEdu(delta_u, t_E, mass)
	return p_xvt(x, vt, mass)*rho_halo(x)/mass*x*x*f_vt(vt)*np.abs(jacobian(delta_u, t_E, mass))

resolution = 1000
vT = np.linspace(-600, 600, resolution)
x = np.linspace(0, 1, resolution)
plt.imshow(pdf_xvt(x[None, :], vT[:, None], mass=60), extent=[x[0], x[-1], vT[0], vT[-1]], aspect='auto', origin='lower', cmap='inferno')
plt.show()