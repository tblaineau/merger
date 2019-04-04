import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time

from scipy.stats import rv_continuous
from scipy.integrate import dblquad
from scipy.misc import derivative

from mpl_toolkits.mplot3d import Axes3D

from astropy import constants
from astropy import units

COLOR_FILTERS = {
	'red_E':{'mag':'red_E', 'err': 'rederr_E'},
	'red_M':{'mag':'red_M', 'err': 'rederr_M'},
	'blue_E':{'mag':'blue_E', 'err': 'blueerr_E'},
	'blue_M':{'mag':'blue_M', 'err': 'blueerr_M'}
}

WORKING_DIR_PATH = "/Volumes/DisqueSauvegarde/working_dir/"

PERIOD_EARTH = 365.2422
alphaS = 80.8941667*np.pi/180.
deltaS = -69.7561111*np.pi/180.
epsilon = (90. - 66.56070833)*np.pi/180.
t_origin = 51442 #(21 septembre 1999) #58747 #(21 septembre 2019)

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
	# t1 = u0*u0
	# t2 = ((t-t0)/tE)**2
	# t3 = delta_u**2 * (np.cos(phi)**2 + (np.sin(phi) * np.cos(beta))**2)
	# t4 = -delta_u * (t-t0)/tE * (np.cos(theta) * np.cos(phi) + np.cos(beta) * np.sin(theta) * np.sin(phi))
	# t5 = u0 * delta_u * (np.sin(theta) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.cos(beta))
	# u = np.sqrt(t1+t2+t3+t4+t5)
	u = np.linalg.norm(u_D-u_t, axis=0)
	return (u**2+2)/(u*np.sqrt(u**2+4))

def microlens(t, params):
	mag, blend, u0, t0, tE, delta_u, theta = params
	return - 2.5*np.log10(blend*np.power(10, mag/-2.5) + (1-blend)*np.power(10, mag/-2.5) * parallax(t, mag, u0, t0, tE, delta_u, theta))

def microlens_simple(t, params):
	mag, blend, u0, t0, tE, delta_u, theta = params
	u = np.sqrt(u0*u0 + ((t-t0)**2)/tE/tE)
	A = (u**2+2)/(u*np.sqrt(u**2+4))
	return - 2.5*np.log10(blend*np.power(10, mag/-2.5) + (1-blend)*np.power(10, mag/-2.5) * A)

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

	return rho_0*A/((x*r_lmc)**2-2*x*r_lmc*B+A)

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

	return np.where(delta_u==0, 0, rho_halo_pdf(g_inverse(delta_u))/np.abs(g_prime(g_inverse(delta_u))))

time_range = np.linspace(48928, 52697, 10000)
t0=50000
tE=500
mag=19
BASE_U0 = 0.8
MAX_DELTA_U = 0.02
delta_u = np.linspace(0,0.03,100)
delta_u2 = delta_u# np.linspace(0,0.06,100)
tE = np.linspace(0, 4000, 100)
params = {
	'mag':mag,
	'blend':0.,
	'u0':BASE_U0,
	't0':t0,
	'tE':tE,
	'delta_u':0.5,	#no parallax
	'theta':10*np.pi/180.
}
params_set = [params['mag'], params['blend'], params['u0'], params['t0'], tE[None,:,None], delta_u2[None, None,:], params['theta']]
st1 = time.time()
absolute_diffs = np.abs(microlens(time_range[:,None,None], params_set)-microlens_simple(time_range[:,None,None], params_set))
print(absolute_diffs.shape)
max_diff = absolute_diffs.mean(axis=0)
print(max_diff.shape)
print(time.time()-st1)

def on_click(event):
	print(event.x, event.y, event.xdata, event.ydata)
	params = {
		'mag':mag,
		'blend':0.,
		'u0':BASE_U0,
		't0':t0,
		'tE':event.ydata,
		'delta_u':event.xdata,
		'theta':90*np.pi/180.
	}
	fig, axs=plt.subplots(2,1,sharex=True)
	axs[0].plot(time_range, microlens(time_range, params.values()))
	axs[0].plot(time_range, microlens_simple(time_range, params.values()))
	axs[0].invert_yaxis()
	axs[1].plot(time_range, microlens(time_range, params.values())-microlens_simple(time_range, params.values()))
	axs[1].invert_yaxis()
	fig.suptitle(r'$t_E = $'+str(event.ydata)+r', $\delta_u = $'+str(event.xdata))
	plt.show()

fig = plt.figure()

plt.imshow(max_diff, origin='lower', interpolation='nearest', cmap='plasma', extent=[delta_u2[0], delta_u2[-1], tE[0], tE[-1]], aspect='auto')
col = plt.colorbar()
#ptE = np.array([tEgen.pdf(x) for x in tE])
#np.save('probte.npy', ptE)
ptE = np.load('probte'+str(MASS)+'m.npy')
prob_field2 = p_delta_u(delta_u)[None,:]*ptE[:,None]
plt.contour(delta_u, tE, prob_field2, levels=4)
#col2 = plt.colorbar()
#print(prob_field2.max(), prob_field2.min())

plt.xlabel(r'$\delta_u$')
plt.ylabel(r'$t_E$')
col.set_label("mean magnitude difference")
fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()