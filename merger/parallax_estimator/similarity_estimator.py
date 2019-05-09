import numpy as np
import pandas as pd
import scipy.optimize
from iminuit import Minuit
import numba as nb
import time

import matplotlib.pyplot as plt
from merger.old.parameter_generator import microlens_parallax, microlens_simple, generate_parameters, delta_u_from_x, tE_from_xvt
from scipy.signal import find_peaks

def distance1(t, params):
	return np.max(np.abs(microlens_simple(t, **params)-microlens_parallax(t, **params)))

def distance2(cnopa, cpara):
	return np.abs(cnopa-cpara).sum()/np.sum(19.-cnopa)

def simplemax_distance(t, params):
	t = np.linspace(params['t0'] - 200, params['t0'] + 200, len(t))
	return np.max(np.abs(np.max(np.abs(microlens_simple(t, **params)-microlens_parallax(t, **params)))))

def max_fitter(t, params):
	return -np.abs((microlens_parallax(t, 19, 0, params['u0'], params['t0'], params['tE'], params['delta_u'], params['theta']) - microlens_simple(t, 19., 0., params['u0'], params['t0'], params['tE'], 0., 0.)))

def fit_simplemax_distance(t, params):
	res = scipy.optimize.differential_evolution(max_fitter, bounds=[(params['t0'] - 400, params['t0'] + 400)], disp=False, popsize=40, mutation=(0.5, 1.0),
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
	tmin = 48928
	tmax = 52697
	t = np.linspace(tmin, tmax, time_sampling)
	ds = []
	c=0
	for params in params_set:
		c+=1
		print(c)
		del params['mass']
		del params['x']
		del params['vt']
		params['mag']=19.
		params['blend']=0.
		ds.append(distance(t, params))
	return ds


def compute_distances(output_name, distance, mass, nb_samples=None, start=None, end=None, seed=1234567890):
	all_params=[]
	np.random.seed(seed)
	all_xvts = np.load('../test/xvt_samples.npy')
	idx = np.arange(0, len(all_xvts)-1)
	np.random.shuffle(idx)
	if nb_samples is None:
		all_xvts = all_xvts[idx][start:end]
	else:
		all_xvts = all_xvts[idx][:nb_samples]
	for g in all_xvts:
		all_params.append(generate_parameters(mass=mass, x=g[0], vt=g[1]))
	df = pd.DataFrame.from_records(all_params)

	st1 = time.time()
	ds = compute_distance(all_params, distance=distance, time_sampling=1000)
	print(time.time()-st1)

	df = df.assign(distance=ds)
	df.to_pickle(output_name)

# all_params=[]
# np.random.seed(1234567890)
# all_xvts = np.load('../test/xvt_samples.npy')
# idx = np.arange(0, len(all_xvts)-1)
# np.random.shuffle(idx)
# print(all_xvts[idx[0]])
# all_xvts = all_xvts[idx]
# plt.hist2d(delta_u_from_x(all_xvts[:,0], mass=60.), tE_from_xvt(all_xvts[:,0], all_xvts[:,1], mass=60.), bins=300, range=((0, 0.05), (0, 1000)))
# plt.show()
# all_xvts = all_xvts[:1]
# plt.hist2d(delta_u_from_x(all_xvts[:,0], mass=60.), tE_from_xvt(all_xvts[:,0], all_xvts[:,1], mass=60.), bins=300, range=((0, 0.05), (0, 1000)))
# plt.show()
# for mass in np.geomspace(0.1, 1000, 5):
# 	for g in all_xvts:
# 		all_params.append(generate_parameters(mass=mass, x=g[0], vt=g[1]))
# df = pd.DataFrame.from_records(all_params)
#
#
# st1 = time.time()
# ds = compute_distance(all_params, distance=minmax_distance_scipy, time_sampling=1000)
# print(time.time()-st1)
#
# df = df.assign(distance=ds)
# df.to_pickle('temp_fittest.pkl')