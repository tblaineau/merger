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

df = pd.read_pickle('chi2.pkl')
df[['distance', 'fitted_params', 'ndof', 'maxdiff']] = pd.DataFrame(df.distance.values.tolist(), index=df.index)

df2 = pd.read_pickle('scipyminmax.pkl')
df2[['distance', 'fitted_params']] = pd.DataFrame(df2.distance.values.tolist(), index=df2.index)
df2.loc[:,'distance'] = df2.distance.map(lambda x: x[0] if isinstance(x, np.ndarray) else x)

print(len(df2))
df.sort_values('idx', inplace=True)
df2.sort_values('idx', inplace=True)
df.set_index('idx', inplace=True)
df2.set_index('idx', inplace=True)
df = df.join(df2, rsuffix='', lsuffix='_chi2').dropna()
print(len(df))

print(df.iloc[0])

fig = plt.figure()
df.plot.scatter('distance', 'maxdiff', s=10*(72./fig.dpi)**2, c='tE', cmap='PiYG', edgecolor='black', vmin=-2000, vmax=2000)
plt.plot(df['distance'], df['distance'], lw=0.5, c='black')
plt.show()

print(df.mass.unique())

tmin = 48928
tmax = 52697
df = df[(df.tE.abs()>15)]
p2 = df.sort_values(by='maxdiff', ascending=False).iloc[0].to_dict()
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
l1 = axs[0].axvline(tmin, color='red', linewidth=0.5)
l2 = axs[0].axvline(tmax, color='red', linewidth=0.5)
pts, = axs[1].plot([], [], ls='', marker='+')
# plt.xlim(51500, 52500)
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

def curvefit_func(t, u0, t0, tE, pu0, pt0, ptE, pdu, ptheta):
	return (microlens_parallax(t, 19, 0, pu0, pt0, ptE, pdu, ptheta) - microlens_simple(t, 19., 0., u0, t0, tE, 0., 0.))**2

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

def curvefit_minuit(params, time_interval=3):
	l1.set_xdata(params['t0']-3*abs(params['tE']))
	l2.set_xdata(params['t0']+3*abs(params['tE']))
	t = np.arange(params['t0']-3*abs(params['tE']), params['t0']+3*abs(params['tE']), time_interval)
	def minuit_wrap(u0, t0, tE):
		r = curvefit_func(t, u0, t0, tE, params['u0'], params['t0'], params['tE'], params['delta_u'], params['theta'])
		update_plot(u0, t0, tE, r.max())
		pts.set_xdata(t)
		pts.set_ydata(r)
		return r.sum()*time_interval

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
			   print_level=1
			   )
	m.migrad()
	return m.get_fmin().fval/(len(t)-3)


import numba as nb

@nb.njit
def drydiff(t, u0, t0, tE, pu0, pt0, ptE, pdu, ptheta):
	return microlens_parallax(t, 19, 0, pu0, pt0, ptE, pdu, ptheta) - microlens_simple(t, 19., 0., u0, t0, tE, 0., 0.)


def absdiff3(t, u0, t0, tE, pu0, pt0, ptE, pdu, ptheta):
	if not isinstance(t, np.ndarray):
		t = np.array([t])
	return (drydiff(t, u0, t0, tE, pu0, pt0, ptE, pdu, ptheta))**2


import scipy.integrate

def integral_curvefit(params, a=-1000000, b=1000000):
	def minuit_wrap(u0, t0, tE):
		quadargs = (u0, t0, tE, params['u0'], params['t0'], params['tE'], params['delta_u'], params['theta'])
		update_plot(u0, t0, tE, 0)
		return scipy.integrate.quad(absdiff3, a=params['t0']-3*abs(params['tE']), b=params['t0']+3*abs(params['tE']), args=quadargs)[0]

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
	print(m.get_fmin().fval)
	return [m.get_fmin().fval, dict(m.values)]

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


# m = Minuit(fitter_minmax,
# 		   u0=p1['u0']+0.1,
# 		   t0=p1['t0'],
# 		   tE=p1['tE'],
# 		   error_u0=0.2,
# 		   error_t0=50,
# 		   error_tE=10,
# 		   limit_u0=(0, 2),
# 		   limit_tE=(p1['tE']*(1-np.sign(p1['tE'])*0.5), p1['tE']*(1+np.sign(p1['tE'])*0.5)),
# 		   limit_t0=(p1['t0']-180, p1['t0']+180),
# 		   errordef=1,
# 		   print_level=1
# 		   )

# res=[]
# st1 = time.time()
# res.append(scipy.optimize.differential_evolution(fitter_minmax,
# 			bounds=[(0, 2), (p1['t0'] - 360., p1['t0'] + 360), (np.abs(p1['tE'])*-2, np.abs(p1['tE'])*2)],
# 			   disp=False, popsize=40, mutation=(0.5, 1.0), strategy='currenttobest1bin', recombination=0.9, maxiter=100))
# res = res[0]
# print(time.time()-st1)
# print(res.fun)

res=[]

def onclick(event):
	global res
	res = integral_curvefit(p1)
	# m.migrad()

cid = fig.canvas.mpl_connect('key_press_event', onclick)
plt.show()

for sigma in [0.01, 0.05, 0.1]:
	print(res/sigma**2)

# cnopa2 = microlens_simple(t, 19., 0., res.x[0], res.x[1], res.x[2], 0., 0.)
# cnopa2 = microlens_simple(t, 19., 0., m.values['u0'], m.values['t0'], m.values['tE'], 0., 0.)
# cnopa2 = microlens_simple(t, 19., 0., res[0], res[1], res[2], 0., 0.)
cnopa2 = microlens_simple(t, 19., 0., res['u0'], res['t0'], res['tE'], 0., 0.)

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
axs[0].plot(t, cnopa2, label='corrected sun')
axs[0].plot(t, cnopa, ls=':', label='sun')
# axs[0].axvline(p2['fitted_params']['t'])
# axs[1].axvline(p2['fitted_params']['t'])
axs[0].plot(t, microlens_simple(t, p2['mag'], p2['blend'], p2['fitted_params'][0], p2['fitted_params'][1], p2['fitted_params'][2], 0, 0), label='DE minimized', color='red', ls='--')
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