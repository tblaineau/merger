import numpy as np
import pandas as pd
import scipy.optimize
from iminuit import Minuit
import numba as nb
import time

import matplotlib.pyplot as plt
from merger.old.parameter_generator import microlens_parallax, microlens_simple, generate_parameter_file
from scipy.signal import find_peaks

def distance1(t, params):
	return np.max(np.abs(microlens_simple(t, **params)-microlens_parallax(t, **params)))

def distance2(cnopa, cpara):
	return np.abs(cnopa-cpara).sum()/np.sum(19.-cnopa)

def simplemax_distance(t, params):
	t = np.linspace(params['t0'] - 200, params['t0'] + 200, len(t))
	return np.max(np.abs(np.max(np.abs(microlens_simple(t, **params)-microlens_parallax(t, **params)))))

def fit_simplemax_distance(t, params):
	def max_fitter_dict(t):
		return -np.abs((microlens_parallax(t, 19, 0, params['u0'], params['t0'], params['tE'], params['delta_u'],
										   params['theta']) - microlens_simple(t, 19., 0., params['u0'], params['t0'],
																			   params['tE'], 0., 0.)))
	res = scipy.optimize.differential_evolution(max_fitter_dict, bounds=[(params['t0'] - 400, params['t0'] + 400)], disp=False, popsize=40, mutation=(0.5, 1.0),
	strategy='best1bin')
	return res.fun


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

def peak_distance(t ,params, min_prominence=0., base_mag=19.):
	cnopa = microlens_simple(t, **params)
	cpara = microlens_parallax(t, **params)
	peaks, infos = find_peaks(cpara-base_mag, prominence=min_prominence)
	if len(peaks):
		return len(peaks)#np.max(infos["prominences"])
	else :
		return 0

def scipy_simple_fit_distance(t, init_params):
	def fitter_func(params):
		u0, t0, tE = params
		return np.max(np.abs((cpara - microlens_simple(time_range, 19., 0., u0, t0, tE, 0., 0.))))

	res = scipy.optimize.minimize(fitter_func, x0=[init_params['u0'], init_params['t0'], init_params['tE']], method='Nelder-Mead')
	return res.fun

def max_fitter(t, u0, t0, tE, pu0, pt0, ptE, pdu, ptheta):
	return -np.abs((microlens_parallax(t, 19, 0, pu0, pt0, ptE, pdu, ptheta) - microlens_simple(t, 19., 0., u0, t0, tE, 0., 0.)))

def minmax_distance_minuit(t, init_params):
	def fitter_minmax(u0, t0, tE):
		return - scipy.optimize.differential_evolution(max_fitter, bounds=[(init_params['t0']-400, init_params['t0']+400)], args=(u0, t0, tE, init_params['u0'], init_params['t0'], init_params['tE'], init_params['delta_u'], init_params['theta']), disp=False, popsize=40, mutation=(0.5, 1.0)).fun
	m = Minuit(fitter_minmax,
			   u0=init_params['u0'],
			   t0=init_params['t0'],
			   tE=init_params['tE'],
			   error_u0=0.5,
			   error_t0=100,
			   error_tE=100,
			   limit_u0=(0, 1),
			   limit_tE=(init_params['tE'] * (1 - np.sign(init_params['tE']) * 0.5), init_params['tE'] * (1 + np.sign(init_params['tE']) * 0.5)),
			   limit_t0=(init_params['t0'] - abs(init_params['tE']), init_params['t0'] + abs(init_params['tE'])),
			   errordef=1,
			   print_level=0
			   )
	m.migrad()
	return m.get_fmin()

def minmax_distance_scipy(t, params):
	"""Compute distance by minimizing the maximum difference between parallax curve and no-parallax curve by changing u0, t0 and tE values."""
	def fitter_minmax(g):
		u0, t0, tE = g
		return - scipy.optimize.differential_evolution(max_fitter, bounds=[(params['t0']-400, params['t0']+400)], args=(u0, t0, tE, params['u0'], params['t0'], params['tE'], params['delta_u'], params['theta']),
					disp=False, popsize=40, mutation=(0.5, 1.0)).fun
	res = scipy.optimize.differential_evolution(fitter_minmax, bounds=[(0, 1),
				(params['t0'] - abs(params['tE']), params['t0'] + abs(params['tE'])), (params['tE'] * (1 - np.sign(params['tE']) * 0.5), params['tE'] * (1 + np.sign(params['tE']) * 0.5))],
				disp=False, popsize=10, mutation=(0.5, 1.0), strategy='currenttobest1bin', atol=0.0001, recombination=0.9)
	return [res.fun, res.x]

@nb.njit
def numba_weighted_mean(a, w):
	s = 0
	n = 0
	for i in range(len(a)):
		s+=a[i]*w[i]
		n+=w[i]
	return s/n

#@nb.njit
def compute_distance(params_set, distance, time_sampling=1000):
	"""
	Compute distance between parallax and no-parallax curves corresponding to **param_set** using the *distance* function.

	Parameters
	----------

	params_set : list
		List of dictionaries containing lens event parameters
	distance : function
		Function to compute distance between para and nopa. Take a time_vector and params_set as parameters.
	time_sampling : int
		Time sampling of the time vector between 48928 and 52697

	Returns
	-------
	list : List of the distances corresponding to the parameters
	"""
	tmin = 48928
	tmax = 52697
	t = np.linspace(tmin, tmax, time_sampling)
	ds = []
	c=0
	for params in params_set:
		c+=1
		print(c)
		params = {key: params[key] for key in ['u0', 't0', 'tE','delta_u', 'theta']}
		params['mag']=19.
		params['blend']=0.
		ds.append(distance(t, params))
	return ds


def compute_distances(output_name, distance, parameter_list, nb_samples=None, start=None, end=None):
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
	"""
	if nb_samples is None:
		if start is not None and end is not None:
			parameter_list = parameter_list[start:end]
		elif (start is not None and end is None) or (start is None and end is not None):
			print('Start and end should both be initialized if nb_samples is None.')
			return None
	else:
		parameter_list = parameter_list[:nb_samples]
	df = pd.DataFrame.from_records(parameter_list)

	st1 = time.time()
	ds = compute_distance(parameter_list, distance=distance, time_sampling=1000)
	print(time.time()-st1)

	df = df.assign(distance=ds)
	df.to_pickle(output_name)

# all_xvts = np.load('../test/xvt_samples.npy')
# np.random.shuffle(all_xvts)
# generate_parameter_file('parameters1M.npy', all_xvts, [0.1, 1, 10, 30, 100, 300])

pms = np.load('parameters1M.npy')
compute_distances('simple_max.pkl', simplemax_distance, pms, nb_samples=100000)