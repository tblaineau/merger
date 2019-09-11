import numpy as np
import pandas as pd
import scipy.optimize
import scipy.integrate
from iminuit import Minuit
import numba as nb
import time
import logging

from merger.old.parameter_generator import microlens_parallax, microlens_simple, generate_parameter_file
from scipy.signal import find_peaks

def distance1(t, params):
	return np.max(np.abs(microlens_simple(t, **params)-microlens_parallax(t, **params)))

def distance2(cnopa, cpara):
	return np.abs(cnopa-cpara).sum()/np.sum(19.-cnopa)

def simplemax_distance(params, dt=1):
	t = np.arange(params['t0'] - 200, params['t0'] + 200, dt)
	return np.max(np.abs(np.max(np.abs(microlens_simple(t, **params)-microlens_parallax(t, **params)))))

@nb.njit
def absdiff_dict(t, params):
		return -np.abs((microlens_parallax(t, 19, 0, params['u0'], params['t0'], params['tE'], params['delta_u'],
										   params['theta']) - microlens_simple(t, 19., 0., params['u0'], params['t0'],
																			   params['tE'], 0., 0.)))

@nb.njit
def absdiff(t, u0, t0, tE, pu0, pt0, ptE, pdu, ptheta):
	return -np.abs((microlens_parallax(t, 19, 0, pu0, pt0, ptE, pdu, ptheta) - microlens_simple(t, 19., 0., u0, t0, tE, 0., 0.)))

@nb.njit
def drydiff(t, u0, t0, tE, pu0, pt0, ptE, pdu, ptheta):
	return microlens_parallax(t, 19, 0, pu0, pt0, ptE, pdu, ptheta) - microlens_simple(t, 19., 0., u0, t0, tE, 0., 0.)


@nb.njit
def absdiff2(t, u0, t0, tE, delta_u, theta):
	t = np.array([t])
	return -np.abs((microlens_parallax(t, 19, 0, u0, t0, tE, delta_u, theta) - microlens_simple(t, 19., 0., u0, t0,
																									tE, 0., 0.)))

def absdiff3(t, u0, t0, tE, pu0, pt0, ptE, pdu, ptheta):
	t = np.array([t])
	return (drydiff(t, u0, t0, tE, pu0, pt0, ptE, pdu, ptheta))**2


def fit_simplemax_distance(params):
	res = scipy.optimize.differential_evolution(absdiff_dict, bounds=[(params['t0'] - 400, params['t0'] + 400)], disp=False, popsize=40, mutation=(0.5, 1.0),
	strategy='best1bin', args=params)
	return res.fun


def fastfit_simplemax_distance(params, init_dt=0.5):
	t = np.arange(params['t0'] - 200, params['t0'] + 200, init_dt)
	init_t = t[(-np.abs(microlens_parallax(t, **params)-microlens_simple(t, **params))).argmin()]
	m = Minuit(absdiff2,
			   t = init_t,
			   t0 = params['t0'],
			   u0 = params['u0'],
			   tE = params['tE'],
			   delta_u = params['delta_u'],
			   theta = params['theta'],
			   fix_t0=True,
			   fix_u0=True,
			   fix_tE=True,
			   fix_delta_u=True,
			   fix_theta=True,
			   error_t=10,
			   errordef=1,
			   print_level=0
			   )
	m.migrad()
	return [m.get_fmin().fval, dict(m.values)]


def curvefit(params, time_interval=3):
	if not 3*abs(params['tE'])<400:
		t = np.arange(params['t0']-3*abs(params['tE']), params['t0']+3*abs(params['tE']), time_interval)
	else:
		t = np.arange(params['t0']-400, params['t0']+400, time_interval)

	def minuit_wrap(u0, t0, tE):
		return (drydiff(t, u0, t0, tE, params['u0'], params['t0'], params['tE'], params['delta_u'], params['theta'])**2).sum()

	m = Minuit(minuit_wrap,
			   u0 = params['u0'],
			   t0 = params['t0'],
			   tE = params['tE'],
			   error_u0 = 0.1,
			   error_t0 = 10,
			   error_tE = 10,
			   limit_u0 = (0, 2),
			   limit_t0 = (params['t0']-400, params['t0']+400),
			   errordef=1,
			   print_level=0
			   )
	m.migrad()
	res = scipy.optimize.differential_evolution(absdiff_dict, bounds=[(params['t0'] - 400, params['t0'] + 400)],
												disp=False, popsize=40, mutation=(0.5, 1.0),
												strategy='best1bin', args=params)
	# init_dt=0.5
	# t = np.arange(m.values['t0'] - 200, m.values['t0'] + 200, init_dt)
	# init_t = (np.abs(microlens_parallax(t, **params)
	# 					- microlens_simple(t, params['mag'], params['blend'], m.values['u0'], m.values['t0'], m.values['tE'], params['delta_u'], params['theta']))).max()
	# return [m.get_fmin().fval, dict(m.values), len(t), init_t]

def integral_curvefit(params, epsabs=1e-8):
	if abs(params["tE"])<608.75:
		a = params['t0'] - 3652.5
		b = params['t0'] + 3652.5
	else:
		a = params['t0'] - 6 * abs(params['tE'])
		b = params['t0'] + 6 * abs(params['tE'])

	def minuit_wrap(u0, t0, tE):
		quadargs = (u0, t0, tE, params['u0'], params['t0'], params['tE'], params['delta_u'], params['theta'])
		return scipy.integrate.quad(absdiff3, a, b, args=quadargs, epsabs=epsabs)[0]

	m = Minuit(minuit_wrap,
			   u0 = params['u0'],
			   t0 = params['t0'],
			   tE = params['tE'],
			   error_u0 = 0.1,
			   error_t0 = 10,
			   error_tE = 10,
			   limit_u0 = (0, 3),
			   limit_t0 = (params['t0']-400, params['t0']+400),
			   errordef=1,
			   print_level=0
			   )
	m.migrad()
	return [m.get_fmin().fval, dict(m.values)]


def max_parallax(params):
	def minus_parallax(t):
		t = np.array([t])
		return microlens_parallax(t, 19, 0, params['u0'], params['t0'], params['tE'], params['delta_u'], params['theta'])

	m = Minuit(minus_parallax,
			   t = params["t0"],
			   error_t = 180,
			   errordef = 1,
			   print_level=0)
	m.migrad()
	return [m.get_fmin().fval, dict(m.values)['t']]


# @nb.jit(nopython=True)
# def dtw_distance(cnopa, cpara):
# 	"""
# 	Dynamic Time Warping function
# 	:param cnopa:
# 	:param cpara:
# 	:return:
# 	"""
# 	dtw = list(np.full(shape=(len(cpara), len(cnopa)), fill_value=np.inf))
# 	dtw[0][0] = 0.
# 	for i in range(1, len(cnopa)):
# 		for j in range(1, len(cpara)):
# 			cost = (cnopa[i]-cpara[j])**2
# 			dtw[i][j] = cost #+ np.min([dtw[i][j-1], dtw[i-1][j-1], dtw[i-1][j]])
# 	print(dtw[-1][-1])
# 	return dtw[-1][-1]


# def fastdtw_distance(cnopa, cpara):
# 	"""
# 	Fast Dynamic Time Warping distance
# 	:param cnopa:
# 	:param cpara:
# 	:return:
# 	"""
# 	distance, path = fastdtw.fastdtw(cnopa, cpara, dist=euclidean)
# 	print(distance)
# 	# print(path)
# 	# path = np.array(path)
# 	# plt.plot(cnopa[path[:,0]])
# 	# plt.plot(cpara[path[:,1]])
# 	# plt.plot(cnopa, linestyle=":")
# 	# plt.plot(cpara, linestyle=":")
# 	# plt.gca().invert_yaxis()
# 	# plt.show()
# 	return distance

def count_peaks(params, min_prominence=0., base_mag=19.):
	"""
	Compute the number of peaks in the parallax event curve

	Parameters
	----------
	params : dict
		Dictionary containing the parameters to compute the parallax curve
	min_prominence : float
		Minimum prominence of a peak to be taken into account, in magnitude
	base_mag : float
		Base magnitude of the unlensed source

	Returns
	-------
	int
		Number of peaks detected
	"""
	t = np.arange(params['t0']-2*np.abs(params['tE']), params['t0']+2*np.abs(params['tE']), 1)
	cpara = microlens_parallax(t, **params)
	peaks, infos = find_peaks(cpara-base_mag, prominence=min_prominence)
	if len(peaks):
		return len(peaks)#np.max(infos["prominences"])
	else :
		return 0


def scipy_simple_fit_distance(init_params):
	def fitter_func(params):
		u0, t0, tE = params
		return np.max(np.abs((cpara - microlens_simple(time_range, 19., 0., u0, t0, tE, 0., 0.))))

	res = scipy.optimize.minimize(fitter_func, x0=[init_params['u0'], init_params['t0'], init_params['tE']], method='Nelder-Mead')
	return res.fun


def minmax_distance_minuit(init_params):
	def fitter_minmax(u0, t0, tE):
		return - scipy.optimize.differential_evolution(absdiff, bounds=[(init_params['t0']-400, init_params['t0']+400)], args=(u0, t0, tE, init_params['u0'], init_params['t0'], init_params['tE'], init_params['delta_u'], init_params['theta']), disp=False, popsize=40, mutation=(0.5, 1.0)).fun
	m = Minuit(fitter_minmax,
			   u0=init_params['u0'],
			   t0=init_params['t0'],
			   tE=init_params['tE'],
			   error_u0=0.5,
			   error_t0=100,
			   error_tE=100,
			   limit_u0=(0, 2),
			   limit_tE=(init_params['tE'] * (1 - np.sign(init_params['tE']) * 0.5), init_params['tE'] * (1 + np.sign(init_params['tE']) * 0.5)),
			   limit_t0=(init_params['t0'] - abs(init_params['tE']), init_params['t0'] + abs(init_params['tE'])),
			   errordef=1,
			   print_level=0
			   )
	m.migrad()
	return m.get_fmin()


def minmax_distance_scipy(params):
	"""Compute distance by minimizing the maximum difference between parallax curve and no-parallax curve by changing u0, t0 and tE values."""
	def fitter_minmax(g):
		u0, t0, tE = g
		return - scipy.optimize.differential_evolution(absdiff, bounds=[(params['t0']-400, params['t0']+400)], args=(u0, t0, tE, params['u0'], params['t0'], params['tE'], params['delta_u'], params['theta']),
					disp=False, popsize=40, mutation=(0.5, 1.0)).fun

	res = scipy.optimize.differential_evolution(fitter_minmax, bounds=[(0, 2),
				(params['t0'] - 400, params['t0'] + 400), (params['tE'] * (1 - np.sign(params['tE']) * 0.5), params['tE'] * (1 + np.sign(params['tE']) * 0.5))],
				disp=False, popsize=40, mutation=(0.5, 1.0), strategy='currenttobest1bin', recombination=0.9)
	return [res.fun, res.x]

def minmax_distance_scipy2(params, time_sampling=0.5, pop_size=40):
	t = np.arange(params['t0'] - 400, params['t0'] + 400, time_sampling)

	def fitter_minmax(g):
		u0, t0, tE = g
		return (drydiff(t, u0, t0, tE, params['u0'], params['t0'], params['tE'], params['delta_u'], params['theta'])**2).max()

	bounds = [(0, 2), (params['t0'] - 400, params['t0'] + 400), (params['tE'] * (1 - np.sign(params['tE']) * 0.5), params['tE'] * (1 + np.sign(params['tE']) * 0.5))]
	init_pop = np.array([np.random.uniform(bounds[0][0], bounds[0][1], pop_size),
				np.random.uniform(bounds[1][0], bounds[1][1], pop_size),
				np.random.uniform(bounds[2][0], bounds[2][1], pop_size),
				]).T

	init_pop[0] = [params['u0'], params['t0'], params['tE']]

	res = scipy.optimize.differential_evolution(fitter_minmax, bounds=bounds, disp=False,
												mutation=(0.5, 1.0), strategy='currenttobest1bin', recombination=0.9,
												init=init_pop)
	return [np.sqrt(res.fun), res.x]

@nb.njit
def numba_weighted_mean(a, w):
	s = 0
	n = 0
	for i in range(len(a)):
		s+=a[i]*w[i]
		n+=w[i
		]
	return s/n


def compute_distances(output_name, distance, parameter_list, nb_samples=None, start=None, end=None, **distance_args):
	"""
	Compute distance between parallax and no-parallax curves using the *distance* function.

	Parameters
	----------

	output_name : str
		Name of the pandas pickle file where is stocked a Dataframe containing the parameters with the associated distance
	distance : function
		Function used to compute the distance between parallax curve and no-parallax curve, with same parameters
	parameter_list : list
		List of lens event parameters
	nb_samples : int
	 	Compute the distance for the **nb_samples** first parameters sets. If nb_samples, start and end are None, compute distance for all parameters.
	start : int
	 	If nb_samples is None, the index of parameter_list from which to computing distance
	end : int
		If nb_samples is None, the index of parameter_list where to stop computing distance
	**distance_args : distance arguments
		Arguments to pass to distance function.
	"""
	if nb_samples is None:
		if start is not None and end is not None:
			parameter_list = parameter_list[start:end]
		elif (start is not None and end is None) or (start is None and end is not None):
			logging.error('Start and end should both be initialized if nb_samples is None.')
			return None
	else:
		parameter_list = parameter_list[:nb_samples]
	df = pd.DataFrame.from_records(parameter_list)

	st1 = time.time()

	ds = []
	i=0
	for params in parameter_list:
		i+=1
		params = {key: params[key] for key in ['u0', 't0', 'tE', 'delta_u', 'theta']}
		params['mag'] = 19.
		params['blend'] = 0.
		st2 = time.time()
		ds.append(distance(params, **distance_args))
		# logging.debug(f'{time.time()-st2} s')
		if i%100 == 0:
			logging.debug(i)

	logging.info(f'{len(parameter_list)} distances computed in {time.time()-st1:.2f} seconds.')

	df = df.assign(distance=ds)
	df.to_pickle(output_name)


logging.basicConfig(level=logging.DEBUG)

# st = time.time()
# pms = np.load('params1M_0.npy', allow_pickle=True)[:1000]
# print(len(pms))
# end = time.time()
# logging.info(f'{len(pms)} parameters loaded in {end-st:.2f} seconds.')
#
# df = pd.DataFrame.from_records(pms)

# df2 = pd.read_pickle('scipyminmax.pkl')['idx']
#
# df = df.merge(df2, on='idx', suffixes=('',''))
# print(len(df))

# print(df.idx.sort_values())
# pms = df.to_records()
#
#
# compute_distances('trash.pkl', integral_curvefit, pms, nb_samples=1000)

"""
logging.debug('Loading xvt_samples')
all_xvts = np.load('../test/xvt_thick_disk.npy')
logging.debug('Done')
logging.debug('Shuffling')
np.random.shuffle(all_xvts)
logging.debug('Done')
logging.debug('Generating parameters sets')
pms = generate_parameter_file('parameters_u02f_TD', all_xvts[:100000], [0.1, 1, 10, 30, 100, 300])

pms = np.load('parameters_u02f_TD.npy', allow_pickle=True)
print(len(pms))
for idx, a in enumerate(np.split(np.array(pms), 20)):
	np.save('parameters_u02f_TD_'+str(idx), a, allow_pickle=True)"""
