import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from iminuit import Minuit
import scipy.optimize

from merger.old.parameter_generator import microlens_parallax, microlens_simple

# df = pd.read_pickle('nbpeaks.pkl')

# df = pd.read_pickle('fast_simplemax.pkl')
# df[['distance', 'fitted_params']] = pd.DataFrame(df.distance.values.tolist(), index=df.index)
# df.distance = df.distance.abs()

df = pd.read_pickle('scipyminmax.pkl')
df[['distance', 'fitted_params']] = pd.DataFrame(df.distance.values.tolist(), index=df.index)
df.loc[:,'distance'] = df.distance.map(lambda x: x[0] if isinstance(x, np.ndarray) else x)

print(df.mass.unique())

tmin = 48928
tmax = 52697
df = df[(df.tE.abs()>15)]
p2 = df.sort_values(by='distance', ascending=False).iloc[100].to_dict()
# p1 = df.iloc[np.random.randint(0, len(df))].to_dict()
print(p2)
p2['blend']=0.
p2['mag']=19.

p1 = {key: p2[key] for key in ['mag', 'blend', 'u0', 'tE', 't0', 'theta', 'delta_u']}

t = np.linspace(tmin, tmax, 10000)
cnopa = microlens_simple(t, **p1)
cpara = microlens_parallax(t, **p1)

fig, axs = plt.subplots(ncols=1, nrows=2, sharex='col')
pdif1, = axs[1].plot(t, np.abs((microlens_parallax(t, 19, 0, p1['u0'], p1['t0'], p1['tE'], p1['delta_u'],p1['theta']) - microlens_simple(t, 19., 0., p1['u0'], p1['t0'], p1['tE'], 0., 0.))))
ppar1, = axs[0].plot(t, -(microlens_parallax(t, 19, 0, p1['u0'], p1['t0'], p1['tE'], p1['delta_u'], p1['theta'])))
pnop1, = axs[0].plot(t, -(microlens_simple(t, 19, 0, p1['u0'], p1['t0'], p1['tE'], p1['delta_u'], p1['theta'])))
axs[0].plot(t, -(microlens_simple(t, 19, 0, p1['u0'], p1['t0'], p1['tE'], p1['delta_u'], p1['theta'])), ls='--')
hl = axs[1].axhline(0, color='black', linewidth=0.5)
hl2 = axs[1].axhline(0, color='red', linewidth=0.5)
plt.xlim(51500, 52500)
curr_max=-np.inf
i=0
explored_parameters=[]


def update_plot(u0, t0, tE, r):
	global curr_max
	global i
	i+=1
	axs[1].set_title(str(i))
	pdif1.set_ydata(np.abs((microlens_parallax(t, 19, 0, p1['u0'], p1['t0'], p1['tE'], p1['delta_u'],p1['theta']) - microlens_simple(t, 19., 0., u0, t0, tE, 0., 0.))))
	ppar1.set_ydata(-(microlens_parallax(t, 19, 0, p1['u0'], p1['t0'], p1['tE'], p1['delta_u'], p1['theta'])))
	pnop1.set_ydata(-(microlens_simple(t, 19, 0, u0, t0, tE, p1['delta_u'], p1['theta'])))
	plt.pause(0.000001)
	hl2.set_ydata(-r)
	if r>curr_max:
		curr_max = r
		hl.set_ydata(-r)
	axs[1].relim()
	axs[1].autoscale_view()
	fig.canvas.draw()


def max_fitter(t, u0, t0, tE, pu0, pt0, ptE, pdu, ptheta):
	return -np.abs((microlens_parallax(t, 19, 0, pu0, pt0, ptE, pdu, ptheta) - microlens_simple(t, 19., 0., u0, t0, tE, 0., 0.)))

def fitter_minmax(u0, t0, tE):
	# u0, t0, tE = params
	res = scipy.optimize.differential_evolution(max_fitter, bounds=[(p1['t0']-400, p1['t0']+400)], args=(u0, t0, tE, p1['u0'], p1['t0'], p1['tE'], p1['delta_u'], p1['theta']), disp=False, popsize=40, mutation=(0.5, 1.0), strategy='best1bin')
	# tm = Minuit(max_fitter, t=t0, error_t=100, limit_t=(tmin, tmax) ,errordef=1, print_level=0)
	# tm.migrad()
	if isinstance(res.fun, np.ndarray):
		r = res.fun[0]
	else:
		r = res.fun
	explored_parameters.append([u0, t0, tE, r])
	update_plot(u0, t0, tE, r)
	return -r

def fitter_minmax_minuit(u0, t0, tE):
	def max_fitter(t):
		t = np.array([t])
		return -np.abs((microlens_parallax(t, 19, 0, p1['u0'], p1['t0'], p1['tE'], p1['delta_u'], p1['theta']) - microlens_simple(t, 19., 0., u0, t0, tE, 0., 0.)))
	tm = Minuit(max_fitter, t=t0, error_t=100, limit_t=(tmin, tmax), errordef=1, print_level=0)
	tm.migrad()
	# update_plot(u0, t0, tE, tm.fval)
	return tm.fval


def microlens_parallax_inv(t, u0, t0, tE, delta_u, theta): return -microlens_parallax(np.array([t]), 19., 0.,  u0, t0, tE, delta_u, theta)
mmin = Minuit(microlens_parallax_inv,
			  t=p1['t0'],
			  u0=p1['u0'],
			  t0=p1['t0'],
			  tE=p1['tE'],
			  delta_u=p1['delta_u'],
			  theta=p1['theta'],
			  fix_u0=True,
			  fix_t0=True,
			  fix_tE=True,
			  fix_delta_u=True,
			  fix_theta=True,
			  error_t = 10,
			  limit_t = (p1['t0']-400, p1['t0']+400),
			  errordef=1,
			  print_level=1
			  )

m = Minuit(fitter_minmax,
		   u0=p1['u0']+0.1,
		   t0=p1['t0'],
		   tE=p1['tE'],
		   error_u0=0.2,
		   error_t0=50,
		   error_tE=10,
		   limit_u0=(0, 2),
		   limit_tE=(p1['tE']*(1-np.sign(p1['tE'])*0.5), p1['tE']*(1+np.sign(p1['tE'])*0.5)),
		   limit_t0=(p1['t0']-180, p1['t0']+180),
		   errordef=1,
		   print_level=1
		   )

from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

class CustomDES(DifferentialEvolutionSolver):
	def __init__(self, func, bounds, args=(),
                 strategy='best1bin', maxiter=None, popsize=15,
                 tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None,
                 maxfun=None, callback=None, disp=False, polish=True,
				init='latinhypercube'):
		DifferentialEvolutionSolver.__init__(self, func, bounds, args=(),
                 strategy='best1bin', maxiter=None, popsize=15,
                 tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None,
                 maxfun=None, callback=None, disp=False, polish=True,
				init='latinhypercube')

	def init_population_lhs(self):
		"""
		Initializes the population with Latin Hypercube Sampling.
		Latin Hypercube Sampling ensures that each parameter is uniformly
		sampled over its range.
		"""
		rng = self.random_number_generator

		# Each parameter range needs to be sampled uniformly. The scaled
		# parameter range ([0, 1)) needs to be split into
		# `self.num_population_members` segments, each of which has the following
		# size:
		segsize = 1.0 / self.num_population_members

		# Within each segment we sample from a uniform random distribution.
		# We need to do this sampling for each parameter.
		samples = (segsize * rng.random_sample(self.population_shape)

				   # Offset each segment to cover the entire parameter range [0, 1)
				   + np.linspace(0., 1., self.num_population_members,
								 endpoint=False)[:, np.newaxis])

		# Create an array for population of candidate solutions.
		self.population = np.zeros_like(samples)

		# Initialize population of candidate solutions by permutation of the
		# random samples.
		for j in range(self.parameter_count):
			order = rng.permutation(range(self.num_population_members))
			self.population[:, j] = samples[order, j]

		self.population[0] = (p1['u0'] - self.__scale_arg1[0]) / self.__scale_arg2[0]

# solv1 = CustomDES(fitter_minmax,
# 			bounds=[(0, 2), (p1['t0'] - 360., p1['t0'] + 360), (np.abs(p1['tE'])*-2, np.abs(p1['tE'])*2)],
# 			   disp=False, popsize=40, mutation=(0.5, 1.0), strategy='currenttobest1bin', recombination=0.9, maxiter=100)

# res=[]
# st1 = time.time()
# res.append(scipy.optimize.differential_evolution(fitter_minmax,
# 			bounds=[(0, 2), (p1['t0'] - 360., p1['t0'] + 360), (np.abs(p1['tE'])*-2, np.abs(p1['tE'])*2)],
# 			   disp=False, popsize=40, mutation=(0.5, 1.0), strategy='currenttobest1bin', recombination=0.9, maxiter=100))
# res = res[0]
# print(time.time()-st1)
# print(res.fun)

def onclick(event):
	m.migrad()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# cnopa2 = microlens_simple(t, 19., 0., res.x[0], res.x[1], res.x[2], 0., 0.)
cnopa2 = microlens_simple(t, 19., 0., m.values['u0'], m.values['t0'], m.values['tE'], 0., 0.)

from mpl_toolkits.mplot3d import Axes3D

# explored_parameters = np.array(explored_parameters)
# print(explored_parameters.shape)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(explored_parameters[:,0], explored_parameters[:,1], explored_parameters[:,3])
# plt.show()
# nrom1 = LogNorm(vmin=abs(explored_parameters[:,3].max()), vmax=0.1)
# fig, axs= plt.subplots(2, 2, sharex='col', sharey='row')
# axs[1, 0].scatter(explored_parameters[:,0], explored_parameters[:,1], c=np.abs(explored_parameters[:,3]), norm=nrom1)
# axs[0, 0].scatter(explored_parameters[:,0], explored_parameters[:,2], c=np.abs(explored_parameters[:,3]), norm=nrom1)
# axs[1, 1].scatter(explored_parameters[:,2], explored_parameters[:,1], c=np.abs(explored_parameters[:,3]), norm=nrom1)
# plt.show()


fig, axs = plt.subplots(nrows=2, ncols=1, sharex='col')
axs[0].plot(t, cpara, label='earth')
# axs[0].plot(t, cnopa2, label='corrected sun')
axs[0].plot(t, cnopa, ls='--', label='sun')
# axs[0].axvline(p2['fitted_params']['t'])
# axs[1].axvline(p2['fitted_params']['t'])
axs[0].plot(t, microlens_simple(t, p2['mag'], p2['blend'], p2['fitted_params'][0], p2['fitted_params'][1], p2['fitted_params'][2], 0, 0), label='minimized', color='orange')
axs[0].legend()
axs[0].invert_yaxis()
axs[1].plot(t, np.abs(cnopa2-cpara), label='minimized', color='orange')
axs[1].plot(t, np.abs(cpara - microlens_simple(t, p2['mag'], p2['blend'], p2['fitted_params'][0], p2['fitted_params'][1], p2['fitted_params'][2], 0, 0)), ls='--', label='DE minimized', color='red')
axs[1].legend()
plt.figure()
nb_bins=100
range_tE = (0, 2000)
range_delta_u = (0, 0.1)
plt.hist2d(np.abs(df['tE']), df['delta_u'], bins=nb_bins, range=(range_tE, range_delta_u))
plt.scatter(np.abs(p1['tE']), p1['delta_u'], marker='x', s=100, color='black')
plt.show()