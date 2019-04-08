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

import geometry

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

MASS=1000

r_lmc = 55000
r_earth = (150*1e6*units.km).to(units.pc).value
r_0 = np.sqrt(4*constants.G/(constants.c**2)*r_lmc*units.pc).decompose([units.Msun, units.pc]).value

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

def r(mass=MASS):
	R_0 = r_0*np.sqrt(mass)
	return r_earth/R_0

t_obs = (10.5*units.year).to(units.s).value
v_T0 = (220*units.km/units.s).to(units.pc/units.s).value

def p_xvt(x, v_T=v_T0, mass=MASS):
	u_lim = 1
	R_E = r_0*np.sqrt(mass*x*(1-x))
	return rho_halo(x)/MASS*r_lmc*(2*u_lim*R_E*t_obs*np.abs(v_T))

def f_vt(v_T, v0=220):
	return (2*v_T/(v0**2))*np.exp(-v_T**2/(v0**2))

def x_from_delta_u(delta_u, mass=MASS):
	return r(mass)**2/(r(mass)**2+delta_u**2)

pc_to_km = (units.pc.to(units.km))

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

def p_tEdu(delta_u, t_E, mass=MASS):
	x = x_from_delta_u(delta_u, mass)
	vt = v_T_from_tEdu(delta_u, t_E, mass)
	return p_xvt(x, mass, vt)*rho_halo(x)/mass*x*x*f_vt(vt)*np.abs(jacobian(delta_u, t_E, mass))

def distance1(time_range, params_set):
	st1 = time.time()
	absolute_diffs = np.abs(microlens(time_range, params_set)-microlens_simple(time_range, params_set))
	print(absolute_diffs.shape)
	max_diff = absolute_diffs.mean(axis=0)
	print(max_diff.shape)
	print(time.time()-st1)
	return max_diff


def distance2(time_range, params_set):
	cpara = microlens_simple(time_range, params_set)
	cnopa = microlens(time_range, params_set)
	diffs = cnopa-cpara
	print(diffs.shape)
	ampli_para = np.abs(diffs.min(axis=0)-diffs.max(axis=0))
	ampli_mulens = np.abs(cnopa.min(axis=0)-cnopa.max(axis=0))
	return ampli_para/ampli_mulens

def distance3(time_range, params_set):
	cpara = microlens_simple(time_range, params_set)
	cnopa = microlens(time_range, params_set)
	diffs = cnopa-cpara
	print(diffs.shape)
	ampli_para = np.mean(np.abs(diffs), axis=0)
	ampli_mulens = np.abs(cnopa.min(axis=0)-cnopa.max(axis=0))
	return ampli_para/ampli_mulens

def distance4(time_range, params_set):
	cpara = microlens_simple(time_range, params_set)/19.
	cnopa = microlens(time_range, params_set)/19.
	diffs = cnopa-cpara
	dt = time_range.flatten()[1]- time_range.flatten()[0]
	print(diffs.shape)
	int_diffs = (np.abs(diffs)).sum(axis=0)
	int_mulens = (cnopa).sum(axis=0)
	return int_diffs/int_mulens	

def distance5(time_range, params_set):
	fixed_params = [params_set[0], params_set[1],params_set[2], params_set[3]]
	det_mat = []
	for tE in params_set[4].flatten():
		for du in params_set[5].flatten():
			cpara = microlens_simple(time_range.flatten(), [*fixed_params, tE, du, params_set[6]])
			cnopa = microlens(time_range.flatten(), [*fixed_params, tE, du, params_set[6]])
			X = np.array([time_range.flatten(), cpara])
			Y = np.array([time_range.flatten(), cnopa])
			mat = geometry.procrustes.best_orthogonal_transform(X, Y)
			nX = mat*X
			det_mat.append(np.abs(nX[1]-Y[1]))
	return np.array(det_mat).reshape((len(params_set[4].flatten()), len(params_set[5].flatten())))

def visualize_parallax_significance(mass=MASS, distance=distance1, u0=0.5, theta=10, cmap='plasma'):
	time_range = np.linspace(48928, 52697, 10000)
	t0=50000
	tE=500
	mag=19
	WIDTH_LENS = 50
	delta_u = np.linspace(0.00001,0.03,WIDTH_LENS)
	delta_u2 = delta_u# np.linspace(0,0.06,100)
	tE = np.linspace(0.00001, 4000, WIDTH_LENS)
	params = {
		'mag':mag,
		'blend':0.,
		'u0':u0,
		't0':t0,
		'tE':tE,
		'delta_u':0.5,	#no parallax
		'theta':theta*np.pi/180.
	}
	params_set = [params['mag'], params['blend'], params['u0'], params['t0'], tE[None,:,None], delta_u2[None, None,:], params['theta']]
	max_diff = distance(time_range[:,None,None], params_set)

	def on_click(event):
		print(event.x, event.y, event.xdata, event.ydata)
		params = {
			'mag':mag,
			'blend':0.,
			'u0':u0,
			't0':t0,
			'tE':event.ydata,
			'delta_u':event.xdata,
			'theta':theta*np.pi/180.
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

	plt.imshow(max_diff, origin='lower', interpolation='nearest', cmap=cmap, extent=[delta_u2[0], delta_u2[-1], tE[0], tE[-1]], aspect='auto')
	col = plt.colorbar()
	NN_POINTS = 1000
	delta_u = np.linspace(0.,0.03,NN_POINTS)
	tE = np.linspace(0.001, 4000, NN_POINTS)
	ptEdu = p_tEdu(delta_u[None,:], tE[:,None], mass)
	plt.contour(delta_u, tE, ptEdu, levels=7)

	plt.xlabel(r'$\delta_u$')
	plt.ylabel(r'$t_E$')
	#col.set_label("mean magnitude difference")
	fig.canvas.mpl_connect('button_press_event', on_click)
	plt.show()

def visualize_parallax_significance_3d(u0=0.1, mass=MASS, distance=distance1, theta=10, levels=None, cmap_ctf='viridis'):
	time_range = np.linspace(48928, 52697, 10000)
	t0=50000
	tE=500
	mag=19
	WIDTH_LENS = 20
	delta_u = np.linspace(0.00001,0.03,WIDTH_LENS)
	delta_u2 = delta_u# np.linspace(0,0.06,100)
	tE = np.linspace(0.001, 4000, WIDTH_LENS)
	params = {
		'mag':mag,
		'blend':0.,
		'u0':u0,
		't0':t0,
		'tE':tE,
		'delta_u':0.5,	#no parallax
		'theta':theta*np.pi/180.
	}
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel(r'$\delta_u$')
	ax.set_ylabel(r'$t_E$')
	if isinstance(u0, np.ndarray):
		params['u0'] = u0[None,None,None,:]
		var = u0
		ax.set_zlabel(r'$u_0$')
	elif isinstance(theta, np.ndarray):
		params['theta'] = theta[None,None,None,:]*np.pi/180.
		var=theta
		ax.set_zlabel(r'$\theta$')
	params_set = [params['mag'], params['blend'], params['u0'], params['t0'], tE[None,:,None, None], delta_u2[None, None,:, None], params['theta']]
	max_diff = distance(time_range[:,None,None,None], params_set)
	print(max_diff.shape)

	if not levels:
		levels = np.quantile(max_diff, np.linspace(0.1,0.9,5))
		print(levels)
	norm = plt.Normalize(max_diff.min(), max_diff.max())
	#levels=np.linspace(max_diff.min(), max_diff.max(), 5)
	#levels=[0.1, 0.25]
	#p = ax.scatter(*np.meshgrid(delta_u, tE, u0), c=norm(max_diff.flatten()), cmap='inferno', s=100)
	for idx, var_i in enumerate(var):
		surf_dutE = np.meshgrid(delta_u, tE)
		ax.contour(*surf_dutE, max_diff[:,:,idx], zdir='z', offset=var_i, cmap=cmap_ctf, levels=levels)
	for idx, tEi in enumerate(tE):
		surf1 = np.meshgrid(var, delta_u)
		ax.contour(surf1[1], max_diff[idx,:,:], surf1[0], zdir='y', offset=tEi, cmap=cmap_ctf, levels=levels)
	# for idx, dui in enumerate(delta_u):
	# 	surf1 = np.meshgrid(var, tE)
	# 	ax.contour(max_diff[:,idx,:], surf1[1], surf1[0], zdir='x', offset=dui, cmap='viridis', levels=levels)
	delta_u2 = np.linspace(0.00001,0.03,1000)
	tE2 = np.linspace(0.001, 4000, 1000)
	ax.contourf(*np.meshgrid(delta_u2, tE2), p_tEdu(delta_u2[None,:], tE2[:,None], mass), zdir='z', offset=0, cmap='Greys')
	ax.set_zlim(var.min(),var.max())
	ax.set_xlim(delta_u.min(),delta_u.max())
	#fig.colorbar(p)
	def on_click(event):
		print(event.x, event.y, event.xdata, event.ydata)
		params['tE']=event.ydata,
		params['delta_u'] = event.xdata
		params['u0']=var[None,:]
		fig2 = plt.figure()
		subax = fig2.add_subplot(111, projection='3d')
		time_range1, var1 = np.meshgrid(var, time_range)
		print(time_range1.shape)
		print(microlens(time_range[:,None], params.values()))
		subax.plot_surface(time_range1, var1, microlens(time_range[:,None], params.values()))
		subax.invert_zaxis()
		#fig.suptitle(r'$t_E = $'+str(event.ydata)+r', $\delta_u = $'+str(event.xdata))
		plt.show()
	#fig.canvas.mpl_connect('button_press_event', on_click)
	plt.show()

def visualize_parameter_space(mass_range=np.array([10,30,100,300,1000])):
	NN_POINTS = 1000
	delta_u = np.linspace(0.,0.06,NN_POINTS)
	tE = np.linspace(0.001, 4000, NN_POINTS)
	#fig = plt.figure()
	#ax = fig.add_subplot(111, projection='3d')
	#fig, axs = plt.subplots(len(mass_range), 1)
	axs = plt.gca()
	for idx, mass in enumerate(mass_range):
		ptEdu = p_tEdu(delta_u[None,:], tE[:,None], mass)
		axs.contour(delta_u, tE, ptEdu)
		#norm = plt.Normalize(ptEdu.min(), ptEdu.max())
		#colors = plt.get_cmap('viridis')(norm(ptEdu))
		#rcount, ccount, _ = colors.shape
		#surf = ax.plot_surface(*np.meshgrid(delta_u, tE), mass*np.ones(ptEdu.shape), facecolors=colors, shade=False)
	plt.show()

def visualize_parameter_space_3d(mass_range=np.array([10,30,100,300,1000]), scale=None):
	NN_POINTS = 1000
	if not scale:
		scale=lambda x:x
	delta_u = np.linspace(0.,0.06,NN_POINTS)
	tE = np.linspace(0.001, 4000, NN_POINTS)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for idx, mass in enumerate(mass_range):
		ptEdu = p_tEdu(delta_u[None,:], tE[:,None], mass)
		ax.contour(delta_u, tE, ptEdu, zdir='z', offset=scale(mass))
	ax.set_zlim(scale(mass_range).min(), scale(mass_range).max())
	# ax.set_zscale('log')
	plt.show()

#visualize_parameter_space_3d(np.logspace(1,3,10), scale=np.log)
visualize_parameter_space()

# visualize_parallax_significance(mass=60, distance=distance2, cmap='coolwarm', theta=280)
# visualize_parallax_significance_3d(u0=np.linspace(0.05,1,20), mass=100, distance=distance5, theta=45)#, levels=[0, 0.1,0.25], cmap_ctf='Reds')
# visualize_parallax_significance_3d(u0=0.1, mass=60, distance=distance2, theta=np.linspace(0.,360,40), levels=[0, 0.1,0.25], cmap_ctf='Reds')