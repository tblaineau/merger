import pandas as pd
import numpy as np
import time
from merger.old.parameter_generator import microlens_parallax, microlens_simple

from iminuit import Minuit
import scipy.optimize

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize, LogNorm

def display(dfi, cutoff=0.):
	df = dfi[dfi['distance']>=cutoff]
	print(len(df)/len(dfi))
	fig, axs = plt.subplots(ncols=5, nrows=1, sharey='all')
	scatter_params = {'marker':'+', 's':1}
	hist2d_prams = {'bins':(20,100), 'norm':LogNorm()}
	for idx, curr_muparameter in enumerate(['u0', 'tE', 'delta_u', 'mass']):
		axs[idx].hist2d(df[curr_muparameter], df['distance'], **hist2d_prams)
		axs[idx].set_title(curr_muparameter)

	axs[-1].hist(df['distance'], bins=100, histtype='step', orientation='horizontal')
	axs[-1].set_xscale('log')
	plt.show()

	fig, axs = plt.subplots(ncols=4, nrows=1, sharey='all')
	for idx, curr_muparameter in enumerate(['u0', 'tE', 'delta_u', 'mass']):
		df.loc[:,curr_muparameter+'_bins'], bins = pd.cut(df[curr_muparameter], 20, retbins=True)
		ratios = df[curr_muparameter+'_bins'].value_counts(sort=False)/pd.cut(dfi[curr_muparameter], bins).value_counts(sort=False)
		colormap = plt.cm.magma_r
		color_val = (pd.cut(dfi[curr_muparameter], bins).value_counts(sort=False)/len(dfi)).to_numpy()
		norm = Normalize(vmin=color_val.min(), vmax=color_val.max())
		axs[idx].bar((bins[:-1]+bins[1:])/2., ratios, align='center', width=1*(bins[1]-bins[0]), color='blue', edgecolor='white')#color=colormap(norm(color_val)))
		axs[idx].twinx().bar((bins[:-1]+bins[1:])/2., color_val, align='center', width=0.7*(bins[1]-bins[0]), color='none', edgecolor='orange')
		axs[idx].set_title(curr_muparameter)
	plt.show()

def cuttoffs_scatter_plots(df, cutoffs=[0.01, 0.05, 0.1]):
	for cutoff in cutoffs:
		tdf = df[df['distance']>cutoff]
		print(len(tdf)/len(df))
		#plt.hist2d(tdf['u0'], tdf['delta_u'], range=((0,1), (0,0.06)), bins=100, norm=LogNorm())
		sns.scatterplot(tdf['tE'], tdf['delta_u'], size=tdf['distance'], hue=tdf['distance'])
		# plt.plot([[0,0], [1,1]], ls="--", c=".3")
		plt.show()

def fraction(dfi, cutoff=0., curr_muparameter='mass', bins=20, show_tot=True, condition=True, binfunc=lambda x:x):
	if isinstance(cutoff, list):
		cutoff = np.array(cutoff)
	elif not isinstance(cutoff, np.ndarray):
		cutoff = np.array([cutoff])
	ax = plt.gca()
	lctf = len(cutoff)
	for idx, ct in enumerate(cutoff):
		df = dfi[(dfi['distance']>=ct) & condition].copy()
		print(len(df),len(df) / len(dfi))
		df.loc[:, curr_muparameter + '_bins'], rbins = pd.cut(df[curr_muparameter], bins=binfunc(bins), retbins=True)
		ratios = df[curr_muparameter + '_bins'].value_counts(sort=False) / pd.cut(dfi[curr_muparameter], bins=rbins).value_counts(sort=False)
		color_val = (pd.cut(dfi[curr_muparameter], bins=rbins).value_counts(sort=False) ).to_numpy()
		ax.set_title(curr_muparameter)
		cmp1 = plt.cm.Blues
		norm = Normalize(0,1)
		# ax.xaxis.set_major_locator(plt.MaxNLocator(21))
		ax.bar((bins[:-1] + bins[1:]) / 2., ratios, align='center', width=1 * (bins[1:] - bins[:-1]), color=cmp1(norm(idx/lctf)*0.5+0.25),edgecolor='white', label='cutoff {:.3f}'.format(ct))  # color=colormap(norm(color_val)))
		if show_tot:
			ax.twinx().bar((bins[:-1] + bins[1:]) / 2., color_val, align='center', width=0.7 * (bins[1:] - bins[:-1]),color='none', edgecolor='orange', label='tot_prop')
		ax.set_xticks(bins)
		for label in ax.get_xticklabels():
			label.set_ha("right")
			label.set_rotation(45)
		ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, ticknb: "{:.3f}".format(binfunc(value))))
	plt.legend()
	ax.legend()
	plt.show()

def parameter_space(dfi, params_pt=None):
	dfdist = dfi[dfi.distance>1]
	fig, axs = plt.subplots(nrows=3, ncols=3, sharex='col')
	range_u0 = (0, 1)
	range_tE = (10, 10000)
	range_delta_u = (0, 0.08)
	nb_bins = 200
	nb_levels =3
	ns1, bins1, patches1 = axs[0, 0].hist(dfdist['u0'], bins=nb_bins, histtype='step', range=range_u0)
	ns2, bins2, patches2 = axs[0, 0].hist(df['u0'], bins=nb_bins, histtype='step', range=range_u0, color='orange')
	axs[1, 0].hist2d(dfi['u0'], dfi['tE'], bins=nb_bins, range=(range_u0, range_tE))
	# axs[1, 0].scatter(dfdist['u0'], dfdist['tE'], marker='.', s=0.1)
	sns.kdeplot(dfdist['u0'], dfdist['tE'], clip=(range_u0, range_tE), ax=axs[1, 0], levels = nb_levels)
	axs[1, 1].hist(dfdist['tE'], bins=nb_bins, histtype='step', range=range_tE)
	axs[1, 1].hist(df['tE'], bins=nb_bins, histtype='step', range=range_tE, color='orange')
	axs[2, 0].hist2d(dfi['u0'], dfi['delta_u'], bins=nb_bins, range=(range_u0, range_delta_u))
	# axs[2, 0].scatter(dfdist['u0'], dfdist['delta_u'], marker='.',  s=0.1)
	sns.kdeplot(dfdist['u0'], dfdist['delta_u'], clip=(range_u0, range_delta_u), ax=axs[2, 0], levels = nb_levels)
	axs[2, 2].hist(dfdist['delta_u'], bins=nb_bins, histtype='step', range=range_delta_u)
	axs[2, 2].hist(df['delta_u'], bins=nb_bins, histtype='step', range=range_delta_u, color='orange')
	axs[2, 1].hist2d(dfi['tE'], dfi['delta_u'], bins=nb_bins, range=(range_tE, range_delta_u))
	# axs[2, 1].scatter(dfdist['tE'], dfdist['delta_u'], marker='.',  s=0.1)
	sns.kdeplot(dfdist['tE'], dfdist['delta_u'], clip=(range_tE, range_delta_u), ax=axs[2, 1], levels = nb_levels)
	if params_pt:
		axs[1, 0].scatter(params_pt['u0'], params_pt['tE'], marker='x', s=100, color='black')
		axs[2, 0].scatter(params_pt['u0'], params_pt['delta_u'], marker='x', s=100, color='black')
		axs[2, 1].scatter(params_pt['tE'], params_pt['delta_u'], marker='x', s=100, color='black')
	plt.show()

#

df = pd.read_pickle('temp_max.pkl')
print(df.iloc[0].distance)
# df = df.join(pd.DataFrame(df.pop('distance').to_list())[['fval', 'is_valid']])
# df.rename(columns={'fval':'distance'}, inplace=True)
#df = df[df.is_valid]


# cutoff_list = [0.01, 0.05, 0.1, 1]
# for mass in np.geomspace(0.1, 1000, 9):
# 	print(mass)
# 	print(len(df[(df.mass==mass) & (df.distance>1)]) / len(df[df.mass==mass]))
# 	print("------------")
# print(len(df))
# cutoff_list = [1, 2]
cutoff_list = [0.01, 0.1, 1]
df = df[df.mass == 100.]


# parameter_space(df[df['mass']==10.])
# parameter_space(df[df['mass']==100.])
# sns.pairplot(df[df.mass==10.], hue='distance', vars=['u0', 'tE', 'delta_u'])

fraction(df, cutoff=cutoff_list, bins=np.linspace(0.9, 3.1, 10), binfunc=lambda x: np.power(10, x), show_tot=True)
fraction(df, curr_muparameter='u0', cutoff=cutoff_list, bins=np.linspace(0, 1, 20), show_tot=False)
fraction(df, curr_muparameter='tE', cutoff=cutoff_list, bins=np.linspace(1, 5, 20), binfunc=lambda x: np.power(10, x), show_tot=True)
fraction(df, curr_muparameter='delta_u', cutoff=cutoff_list, bins=np.linspace(0, 0.2, 20), show_tot=True)
# cuttoffs_scatter_plots(df, cutoffs=[0, 1])
# c1 = df['mass']>0
# plt.hist2d(df[c1]['x'], df[c1]['tE'], bins=300, range=((0, 1), (0, 5000)))
# plt.xlabel(r'$x$')
# plt.ylabel(r'$t_E$')
# plt.show()
# display(df, cutoff=0.)


tmin = 48928
tmax = 52697
p1 = df.sort_values(by='distance', ascending=False).iloc[0].to_dict()
# p1 = df.iloc[np.random.randint(0, len(df))].to_dict()
print(p1)
p1['blend']=0.
p1['mag']=19.
del p1['distance']
del p1['mass']
del p1['x']
del p1['vt']
p1.pop('is_valid', None)
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
explored_parameters=[]


def update_plot(u0, t0, tE, r):
	global curr_max
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
	# update_plot(u0, t0, tE, r)
	return -r

def fitter_minmax_minuit(u0, t0, tE):
	def max_fitter(t):
		t = np.array([t])
		return -np.abs((microlens_parallax(t, 19, 0, p1['u0'], p1['t0'], p1['tE'], p1['delta_u'], p1['theta']) - microlens_simple(t, 19., 0., u0, t0, tE, 0., 0.)))
	tm = Minuit(max_fitter, t=t0, error_t=100, limit_t=(tmin, tmax), errordef=1, print_level=0)
	tm.migrad()
	update_plot(u0, t0, tE, tm.fval)
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
			   disp=False, popsize=10, mutation=(0.5, 1.0), strategy='currenttobest1bin', atol=0.0001, recombination=0.9))
res = res[0]
print(time.time()-st1)
print(res.fun)

# def onclick(event):
# 	m.migrad()

# cid = fig.canvas.mpl_connect('button_press_event', onclick)
# plt.show()

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
axs[0].plot(t, cpara, label='sun')
axs[0].plot(t, cnopa2, label='corrected earth')
axs[0].plot(t, cnopa, label='original earth')
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