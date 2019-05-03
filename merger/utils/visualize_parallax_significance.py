import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import find_peaks, peak_prominences
from astropy import constants
from astropy import units
import time
import numba as nb
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import itertools

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

@nb.njit
def nb_microlens_simple(t, mag, u0, t0, tE):
	if tE==0:
		return 0
	u = np.sqrt(u0*u0 + ((t-t0)**2)/tE/tE)
	if u==0:
		return np.inf
	return mag - 2.5*np.log10((u**2+2)/(u*np.sqrt(u**2+4)))

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
def x_from_delta_u(delta_u, mass):
	return r(mass)**2/(r(mass)**2+delta_u**2)


@nb.njit
def v_T_from_tEdu(delta_u, t_E, mass):
	R_0 = r_0*np.sqrt(mass)*pc_to_km
	ri = r(mass)
	return R_0/(t_E*86400) * ri*delta_u/(ri**2+delta_u**2)


@nb.njit
def delta_u_from_x(x, mass):
	return r(mass)*np.sqrt((1-x)/x)


@nb.njit
def tE_from_xvt(x, vt, mass):
	return r_0 * np.sqrt(mass*x*(1-x)) / (vt*kms_to_pcd)


@nb.njit
def jacobian(delta_u, t_E, mass):
	R_0 = r_0*np.sqrt(mass)*pc_to_km
	ri = r(mass)
	h1 = -2*ri**2*delta_u/(ri**2+delta_u**2)**2
	sqrth2 = ri*delta_u/(ri**2+delta_u**2)
	return -h1*sqrth2*R_0/(t_E*86400)**2


def pdf_tEdu(delta_u, t_E, mass):
	x = x_from_delta_u(delta_u, mass)
	vt = v_T_from_tEdu(delta_u, t_E, mass)
	return p_xvt(x, vt, mass)*f_vt(vt)*np.abs(jacobian(delta_u, t_E, mass))


def distance1(time_range, params_set):
	st1 = time.time()
	absolute_diffs = np.abs(microlens(time_range, params_set)-microlens_simple(time_range, params_set))
	print(absolute_diffs.shape)
	max_diff = absolute_diffs.max(axis=0)
	print(max_diff.shape)
	print(time.time()-st1)
	return max_diff


def distance2(time_range, params_set):
	cnopa = microlens_simple(time_range, params_set)
	cpara = microlens(time_range, params_set)
	diffs = cpara-cnopa
	print(diffs.shape)
	ampli_para = np.abs(diffs.min(axis=0)-diffs.max(axis=0))
	ampli_mulens = np.abs(cnopa.min(axis=0)-cnopa.max(axis=0))
	return ampli_para/ampli_mulens


def distance3(time_range, params_set):
	cnopa = microlens_simple(time_range, params_set)
	cpara = microlens(time_range, params_set)
	diffs = cpara-cnopa
	print(diffs.shape)
	ampli_para = np.mean(np.abs(diffs), axis=0)
	ampli_mulens = np.abs(cnopa.min(axis=0)-cnopa.max(axis=0))
	return ampli_para/ampli_mulens


def distance4(time_range, params_set):
	cnopa = microlens_simple(time_range, params_set)/19.
	cpara = microlens(time_range, params_set)/19.
	diffs = cpara-cnopa
	dt = time_range[1,0,0]- time_range[0,0,0]
	print(diffs.shape)
	int_diffs = (np.abs(diffs)).sum(axis=0)
	int_mulens = (cnopa).sum(axis=0)
	return int_diffs/int_mulens	


def distance5(time_range, params_set, min_prominence=0.05):
	"""
	Distance is number of peak detected with a certain minimum prominence.
	:param time_range:
	:param params_set:
	:return:
	"""
	cpara = microlens(time_range, params_set)
	nb_peaks = []
	cpara_f = cpara.reshape((len(time_range), np.prod(cpara.shape[1:])))
	for c in cpara_f.T:
		peaks, _ = find_peaks(params_set[0]-c, prominence=min_prominence)
		nb_peaks.append(len(peaks))
	r = np.array(nb_peaks)
	print(np.unique(r))
	return r.reshape(cpara.shape[1:])

def scipy_simple_fit_distance(time_range, params_set):
	cpara = microlens(time_range, params_set)
	r=[]
	cpara_f = cpara.reshape((len(time_range), np.prod(cpara.shape[1:]))).T
	time_range = time_range.flatten()
	params_set_f = [list(a) for a in np.broadcast(*params_set)]
	for idx, cp in enumerate(cpara_f):
		pps = params_set_f[idx]
		print(pps)
		print(idx)
		@nb.njit
		def fitter_func(params):
			u0, t0, tE = params
			tot = 0
			for i in range(len(cp)):
				tot += abs(cp[i] - nb_microlens_simple(time_range[i], 19., u0, t0, tE))
			return tot

		res = minimize(fitter_func, x0=[pps[2], pps[3], pps[4]], method='Nelder-Mead')
		r.append(res.fun)
	return np.array(r).reshape(cpara.shape[1:])

def visualize_parallax_significance(mass, distance=distance1, distance_args=[], u0=0.5, theta=10, delta_u_range=(0.00001,0.03), tE_range=(0.00001, 4000), cmap_distance='inferno', cmap_distro='viridis'):
	time_range = np.linspace(48928, 52697, 1000)
	t0=50000
	mag=19
	WIDTH_LENS = 50
	delta_u = np.linspace(*delta_u_range,WIDTH_LENS)
	tE = np.linspace(*tE_range, WIDTH_LENS)
	params = {
		'mag':mag,
		'blend':0.,
		'u0':u0,
		't0':t0,
		'tE':tE,
		'delta_u':0.5,	#no parallax
		'theta':theta*np.pi/180.
	}
	params_set = [params['mag'], params['blend'], params['u0'], params['t0'], tE[None,:,None], delta_u[None, None,:], params['theta']]
	max_diff = distance(time_range[:,None,None], params_set, *distance_args)

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
		fig, axs=plt.subplots(2, 1, sharex=True)

		for cu0 in np.linspace(0, 1, 5):
			params["u0"] = cu0

			cnopa = microlens_simple(time_range, params.values())
			cpara = microlens(time_range, params.values())

			axs[0].plot(time_range, cpara)
			axs[1].plot(time_range, cpara-cnopa)


		#prominences
		peaks, _ = find_peaks(mag-cpara)
		prominences = peak_prominences(mag-cpara, peaks)[0]
		print(prominences)
		contour_heights = cpara[peaks] + prominences
		axs[0].vlines(x=time_range[peaks], ymin=contour_heights, ymax=cpara[peaks])

		axs[0].plot(time_range, cnopa)
		axs[0].invert_yaxis()
		axs[1].invert_yaxis()
		fig.suptitle(r'$t_E = $'+str(event.ydata)+r', $\delta_u = $'+str(event.xdata))
		plt.show()

	fig = plt.figure()

	plt.imshow(max_diff, origin='lower', interpolation='nearest', cmap=cmap_distance, extent=[delta_u[0], delta_u[-1], tE[0], tE[-1]], aspect='auto')
	col = plt.colorbar()
	NN_POINTS = 1000
	ptEdu = pdf_tEdu(delta_u[None,:], tE[:,None], mass)
	plt.contour(delta_u, tE, ptEdu, levels=7, cmap=cmap_distro)

	plt.xlabel(r'$\delta_u$')
	plt.ylabel(r'$t_E$')
	#col.set_label("mean magnitude difference")
	fig.canvas.mpl_connect('button_press_event', on_click)
	plt.show()


def visualize_parallax_significance_xvt(mass, distance=distance1, distance_args=[], u0=0.5, theta=10, x_range=(0.0, 1.0), vt_range=(0., 600), cmap_distance='inferno', cmap_distro='viridis', vmax=None):
	time_range = np.linspace(48928, 52697, 10000)
	t0=50000
	mag=19
	WIDTH_LENS = 50
	x = np.linspace(*x_range, WIDTH_LENS)
	vt = np.linspace(*vt_range, WIDTH_LENS)
	delta_u = delta_u_from_x(x=x[None,:], mass=mass)
	tE = tE_from_xvt(x=x[None,:], vt=vt[:,None], mass=mass)
	params = {
		'mag':mag,
		'blend':0.,
		'u0':u0,
		't0':t0,
		'tE':tE,
		'delta_u':0.5,	#no parallax
		'theta':theta*np.pi/180.
	}
	params_set = [params['mag'], params['blend'], params['u0'], params['t0'], tE[None,:,:], delta_u[None,:,:], params['theta']]
	max_diff = distance(time_range[:,None,None], params_set, *distance_args)

	def on_click(event):
		print(event.x, event.y, event.xdata, event.ydata)
		params = {
			'mag':mag,
			'blend':0.,
			'u0':u0,
			't0':t0,
			'tE':tE_from_xvt(x = event.xdata, vt = event.ydata, mass=mass),
			'delta_u':delta_u_from_x(x = event.xdata, mass=mass),
			'theta':theta*np.pi/180.
		}
		fig, axs=plt.subplots(2,1,sharex=True)

		cnopa = microlens_simple(time_range, params.values())
		cpara = microlens(time_range, params.values())

		#prominences
		peaks, _ = find_peaks(mag-cpara)
		prominences = peak_prominences(mag-cpara, peaks)[0]
		print(prominences)
		contour_heights = cpara[peaks] + prominences
		axs[0].vlines(x=time_range[peaks], ymin=contour_heights, ymax=cpara[peaks])

		axs[0].plot(time_range, cpara)
		axs[0].plot(time_range, cnopa)
		axs[0].invert_yaxis()
		axs[1].plot(time_range, cpara-cnopa)
		axs[1].invert_yaxis()
		fig.suptitle(r'$t_E = $'+str(event.ydata)+r', $\delta_u = $'+str(event.xdata))
		plt.show()

	fig = plt.figure()

	plt.imshow(max_diff, origin='lower', interpolation='nearest', cmap=cmap_distance, extent=[x[0], x[-1], vt[0], vt[-1]], aspect='auto', vmax=vmax)
	col = plt.colorbar()
	NN_POINTS = 1000
	ptEdu = pdf_tEdu(delta_u, tE, mass)
	plt.contour(x, vt, ptEdu, levels=7, cmap=cmap_distro)

	plt.xlabel(r'$x$')
	plt.ylabel(r'$v_T$')
	#col.set_label("mean magnitude difference")
	fig.canvas.mpl_connect('button_press_event', on_click)
	plt.show()


def visualize_parallax_significance_3d(mass, u0=0.1, distance=distance1, distance_args=[], theta=10, delta_u_range=(0.00001,0.03), tE_range=(0.00001, 4000)):
	time_range = np.linspace(48928, 52697, 10000)
	t0=50000
	tE=500
	mag=19
	WIDTH_LENS = 20
	delta_u = np.linspace(*delta_u_range, WIDTH_LENS)
	tE = np.linspace(*tE_range, WIDTH_LENS)
	params = {
		'mag':mag,
		'blend':0.,
		'u0':u0,
		't0':t0,
		'tE':tE,
		'delta_u':0.5,	#no parallax
		'theta':theta*np.pi/180.
	}
	if isinstance(u0, np.ndarray):
		params['u0'] = u0[None,None,None,:]
		var = u0
	elif isinstance(theta, np.ndarray):
		params['theta'] = theta[None,None,None,:]*np.pi/180.
		var=theta
	params_set = [params['mag'], params['blend'], params['u0'], params['t0'], tE[None,:,None, None], delta_u[None, None,:, None], params['theta']]
	max_diff = distance(time_range[:,None,None,None], params_set, *distance_args)
	print(max_diff.shape)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	norm = plt.Normalize(max_diff.min(), max_diff.max())
	levels=[0.1, 0.25]
	levels=[2, 3, 4, 5, 6]
	#p = ax.scatter(*np.meshgrid(delta_u, tE, u0), c=norm(max_diff.flatten()), cmap='inferno', s=100)
	for idx, u0i in enumerate(var):
		surf_dutE = np.meshgrid(delta_u, tE)
		ax.contour(*surf_dutE, max_diff[:,:,idx], zdir='z', offset=u0i, cmap='viridis', levels=levels)
	ax.contourf(*surf_dutE, pdf_tEdu(delta_u[None,:], tE[:,None], mass), zdir='z', offset=0, cmap='inferno')
	ax.set_zlim(var.min(),var.max())
	#fig.colorbar(p)

	plt.show()


def visualize_parameter_space(mass_range=np.array([10,30,100,300,1000])):
	NN_POINTS = 1000
	delta_u = np.linspace(0.,0.06,NN_POINTS)
	tE = np.linspace(0.001, 4000, NN_POINTS)
	#fig = plt.figure()
	#ax = fig.add_subplot(111, projection='3d')
	fig, axs = plt.subplots(len(mass_range), 1)
	for idx, mass in enumerate(mass_range):
		ptEdu = pdf_tEdu(delta_u[None,:], tE[:,None], mass)
		axs[idx].contourf(delta_u, tE, ptEdu)
		#norm = plt.Normalize(ptEdu.min(), ptEdu.max())
		#colors = plt.get_cmap('viridis')(norm(ptEdu))
		#rcount, ccount, _ = colors.shape
		#surf = ax.plot_surface(*np.meshgrid(delta_u, tE), mass*np.ones(ptEdu.shape), facecolors=colors, shade=False)
	plt.show()


def visualize_parameter_space_imshow(mass, nb_points=100, interpolation=None, cmap='viridis'):
	delta_u = np.linspace(0.,0.06,nb_points)
	tE = np.linspace(0.001, 4000, nb_points)
	ptEdu = pdf_tEdu(delta_u[None,:], tE[:,None], mass)
	plt.imshow(ptEdu, origin='lower', interpolation=interpolation, cmap=cmap, extent=[delta_u[0], delta_u[-1], tE[0], tE[-1]], aspect='auto')
	plt.xlabel(r'$\delta_u$')
	plt.ylabel(r'$t_E$')
	plt.show()


def interactive_parameter_space(nb_points=100, cmap='viridis'):
	delta_u = np.linspace(0.,0.06,nb_points)
	tE = np.linspace(0.001, 4000, nb_points)
	ptEdu = pdf_tEdu(delta_u[None,:], tE[:,None], 60)
	fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios':[3,1]})
	def update_imshow(mass):
		ptEdu = pdf_tEdu(delta_u[None,:], tE[:,None], mass)
		col.set_clim(vmin=ptEdu.min(), vmax=ptEdu.max())
		imshw1.set_data(ptEdu)
	imshw1 = axs[0].imshow(ptEdu, origin='lower', cmap=cmap, extent=[delta_u[0], delta_u[-1], tE[0], tE[-1]], aspect='auto')
	col = plt.colorbar(imshw1)
	update_imshow(60)

	bt1 = Slider(axs[1], 'mass', 1, 1000)
	bt1.on_changed(update_imshow)
	plt.show()

# visualize_parameter_space_imshow(100, 1000, cmap='inferno')
# interactive_parameter_space(1000, cmap='inferno')
# visualize_parallax_significance(mass=30, u0=0.3, theta=45, distance=distance5, distance_args=[0.0], cmap_distance='viridis', cmap_distro='inferno')
visualize_parallax_significance(mass=30, u0=0.3, theta=45, distance=scipy_simple_fit_distance, cmap_distance='viridis', cmap_distro='inferno')
# visualize_parallax_significance_xvt(mass=60., u0=0.01, theta=45, vt_range=(0,100.), distance=distance1, distance_args=[], cmap_distance='viridis', cmap_distro='inferno', vmax=0.1)
# visualize_parallax_significance_3d(u0=np.linspace(0.05,1,20), mass=100, distance=distance5, theta=45, distance_args=[0.])

