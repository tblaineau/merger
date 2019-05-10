import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from iminuit import Minuit
import scipy.optimize

from merger.old.parameter_generator import microlens_parallax, microlens_simple

df = pd.read_pickle('temp_max.pkl')


tmin = 48928
tmax = 52697
p1 = df.sort_values(by='distance', ascending=False).iloc[35].to_dict()
# p1 = df.iloc[np.random.randint(0, len(df))].to_dict()
print(p1)
p1['blend']=0.
p1['mag']=19.

del p1['distance']
del p1['mass']
del p1['x']
del p1['vt']
p1.pop('is_valid', None)
p1.pop('fitted_params', None)

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
# plt.xlim(51200, 51600)
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

def fitter_minmax(params):
	u0, t0, tE = params
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


# m = Minuit(fitter_minmax,
# 		   u0=p1['u0'],
# 		   t0=p1['t0'],
# 		   tE=p1['tE'],
# 		   error_u0=0.2,
# 		   error_t0=50,
# 		   error_tE=10,
# 		   limit_u0=(0, 1),
# 		   limit_tE=(p1['tE']*(1-np.sign(p1['tE'])*0.5), p1['tE']*(1+np.sign(p1['tE'])*0.5)),
# 		   limit_t0=(p1['t0']-180, p1['t0']+180),
# 		   errordef=1,
# 		   print_level=1
# 		   )

res=[]
st1 = time.time()
res.append(scipy.optimize.differential_evolution(fitter_minmax,
			bounds=[(0, 1), (p1['t0'] - abs(p1['tE']), p1['t0'] + abs(p1['tE'])), (p1['tE'] * (1 - np.sign(p1['tE']) * 0.5), p1['tE'] * (1 + np.sign(p1['tE']) * 0.5))],
			   disp=False, popsize=40, mutation=(0.5, 1.0), strategy='currenttobest1bin', recombination=0.9, maxiter=100))
res = res[0]
print(time.time()-st1)
print(res.fun)

# def onclick(event):
# 	m.migrad()
#
# cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

cnopa2 = microlens_simple(t, 19., 0., res.x[0], res.x[1], res.x[2], 0., 0.)
# cnopa2 = microlens_simple(t, 19., 0., m.values['u0'], m.values['t0'], m.values['tE'], 0., 0.)

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
axs[0].plot(t, cnopa, ls='--', label='original sun')
axs[0].legend()
axs[0].invert_yaxis()
axs[1].plot(t, np.abs(cnopa2-cpara))
plt.figure()
nb_bins=100
range_tE = (0, 2000)
range_delta_u = (0, 0.1)
plt.hist2d(np.abs(df['tE']), df['delta_u'], bins=nb_bins, range=(range_tE, range_delta_u))
plt.scatter(np.abs(p1['tE']), p1['delta_u'], marker='x', s=100, color='black')
plt.show()