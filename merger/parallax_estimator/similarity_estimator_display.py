import pandas as pd
import numpy as np


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

def compare_distances_df(one, two):
	one.sort_values(by='x', inplace=True)
	two.sort_values(by='x', inplace=True)
	print(one.iloc[0])
	print(two.iloc[0])
	print(len(one))
	print(len(two))
	joined = one.join(two, on='x', lsuffix='_one', rsuffix='_two')
	joined.dropna(axis=0, how='any', subset=['x_two', 'vt_two', 'x_one', 'vt_one'], inplace=True)
	print(len(joined))

# df = pd.read_pickle('fittermax/scipyminmax1000.pkl')
# df[['distance', 'fitted_params']] = pd.DataFrame(df.distance.values.tolist(), index=df.index)
# df.loc[:,'distance'] = df.distance.map(lambda x: x[0] if isinstance(x, np.ndarray) else x)
# print(df.distance)

df = pd.read_pickle('temp_max.pkl')

cmap1 = plt.cm.Blues
norm = Normalize(vmin=0, vmax=1)
df.sort_values(by='mass', inplace=True)
df.mass.value_counts(sort=False).sort_index().plot.bar(color='none', edgecolor='black', width=1)
df[df.distance>0.01].mass.value_counts(sort=False).sort_index().plot.bar(color=cmap1(norm(0.25)), edgecolor='white', width=1)
df[df.distance>0.1].mass.value_counts(sort=False).sort_index().plot.bar(color=cmap1(norm(0.75)), edgecolor='white', width=1)
plt.show()

# df = pd.read_pickle('temp_fittermax1000.pkl')

# df2 = df.join(pd.DataFrame(df.pop('distance').to_list())[['fval', 'is_valid']])
# df2.rename(columns={'fval':'distance'}, inplace=True)
#df = df[df.is_valid]

cutoff_list = [0.001, 0.01, 0.1]


# parameter_space(df[df['mass']==10.])
# parameter_space(df[df['mass']==100.])
# sns.pairplot(df[df.mass==10.], hue='distance', vars=['u0', 'tE', 'delta_u'])

fraction(df, cutoff=cutoff_list, bins=np.linspace(0.9, 3.1, 10), binfunc=lambda x: np.power(10, x), show_tot=True)
fraction(df, curr_muparameter='u0', cutoff=cutoff_list, bins=np.linspace(0, 1, 20), show_tot=False)
fraction(df, curr_muparameter='tE', cutoff=cutoff_list, bins=np.linspace(1, 5, 20), binfunc=lambda x: np.power(10, x), show_tot=True)
fraction(df, curr_muparameter='delta_u', cutoff=cutoff_list, bins=np.linspace(0, 0.2, 20), show_tot=True)
cuttoffs_scatter_plots(df, cutoffs=[0, 0.01, 0.1])
# c1 = df['mass']>0
# plt.hist2d(df[c1]['x'], df[c1]['tE'], bins=300, range=((0, 1), (0, 5000)))
# plt.xlabel(r'$x$')
# plt.ylabel(r'$t_E$')
# plt.show()
# display(df, cutoff=0.)