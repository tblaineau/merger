import numpy as np
import astropy.units as units
import astropy.constants as constants

#fastdtw
import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import seaborn as sns
import time
import numba as nb

from iminuit import Minuit
import scipy.optimize

from scipy.integrate import quad

COLOR_FILTERS = {
	'red_E':{'mag':'red_E', 'err': 'rederr_E'},
	'red_M':{'mag':'red_M', 'err': 'rederr_M'},
	'blue_E':{'mag':'blue_E', 'err': 'blueerr_E'},
	'blue_M':{'mag':'blue_M', 'err': 'blueerr_M'}
}

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
def pdf_xvt(x, vt, mass):
	if x<0 or x>1 or vt<0:
		return 0
	return p_xvt(x, vt, mass)*f_vt(vt)

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

@nb.vectorize([nb.float64(nb.float64, nb.float64, nb.float64)])
def pdf_tEdu(t_E, delta_u, mass):
	x = x_from_delta_u(delta_u, mass)
	vt = v_T_from_tEdu(delta_u, t_E, mass)
	return pdf_xvt(x, vt, mass)*np.abs(jacobian(delta_u, t_E, mass))

@nb.njit
def randomizer(x, vt):
	return np.array([np.random.triangular(x-0.1, x, x+0.1), np.random.triangular(vt-100, vt, vt + 100)])

@nb.njit
def randomizer_gauss(x, vt):
	return np.array([np.random.normal(loc=x, scale=0.1), np.random.normal(loc=vt, scale=300)])

def metropolis_hastings(func, g, nb_samples, start, kwargs={}):
	samples = []
	current_x = start
	accepted=0
	while nb_samples > len(samples):
		proposed_x = g(*current_x)
		tmp = func(*current_x, **kwargs)
		if tmp!=0:
			threshold = min(1., func(*proposed_x, **kwargs) / tmp)
		else:
			threshold = 1
		if np.random.uniform() < threshold:
			current_x = proposed_x
			accepted+=1
		if current_x[0]>0 and current_x[0]<1:
			samples.append(current_x)
	print(accepted, accepted/nb_samples)
	return np.array(samples)

def generate_parameters(mass, seed=None, blending=False, parallax=False, s=None, x=None, vt=None):
	"""
	Parameters to generate : u0, tE, 𝛅u, theta, t0, blends factors
	:param mass:
	:param seed:
	:param blending:
	:return:
	"""
	tmin = 48928.
	tmax = 52697.
	u_max = 1.
	max_blend=0.7

	if seed:
		seed = int(seed.replace('lm0', '').replace('k', '0').replace('l', '1').replace('m', '2').replace('n', '3'))
		np.random.seed(seed)

	u0 = np.random.uniform(0,u_max)
	if not x or not vt:
		if not isinstance(s, np.ndarray):
			s = np.load('../test/xvt_samples.npy')
		x , vt = s[np.random.randint(0, s.shape[0])]
	vt *= np.random.choice([-1., 1.])
	delta_u = delta_u_from_x(x, mass=mass)
	tE = tE_from_xvt(x, vt, mass=mass)
	t0 = np.random.uniform(tmin - tE / 2., tmax + tE / 2.)
	blend_factors = {}
	for key in COLOR_FILTERS.keys():
		if blending:
			blend_factors[key] = np.random.uniform(0, max_blend)
		else:
			blend_factors[key] = 0
	theta = np.random.uniform(0, 2 * np.pi)
	params = {
		'blend':blend_factors,
		'u0':u0,
		't0':t0,
		'tE':tE,
		'delta_u':delta_u,
		'theta':theta,
		'mass':mass,
		'x':x,
		'vt':vt,
	}
	return params


PERIOD_EARTH = 365.2422
alphaS = 80.8941667*np.pi/180.
deltaS = -69.7561111*np.pi/180.
epsilon = (90. - 66.56070833)*np.pi/180.
t_origin = 51442 #(21 septembre 1999) #58747 #(21 septembre 2019)

sin_beta = np.cos(epsilon)*np.sin(deltaS) - np.sin(epsilon)*np.cos(deltaS)*np.sin(alphaS)
beta = np.arcsin(sin_beta) #ok because beta is in -pi/2; pi/2
if abs(beta)==np.pi/2:
	lambda_star = 0
else:
	lambda_star = np.sign((np.sin(epsilon)*np.sin(deltaS)+np.cos(epsilon)*np.sin(alphaS)*np.cos(deltaS))/np.cos(beta)) * np.arccos(np.cos(deltaS)*np.cos(alphaS)/np.cos(beta))

@nb.njit
def microlens_parallax(t_range, mag, blend, u0, t0, tE, delta_u, theta):
	out = np.zeros(t_range.shape)
	for i in range(len(t_range)):
		t = t_range[i]
		tau = (t-t0)/tE
		phi = 2*np.pi * (t-t_origin)/PERIOD_EARTH - lambda_star
		t1 = u0**2 + tau**2
		t2 = delta_u**2 * (np.sin(phi)**2 + np.cos(phi)**2*sin_beta**2)
		t3 = -2*delta_u*u0 * (np.sin(phi)*np.sin(theta) + np.cos(phi)*np.cos(theta)*sin_beta)
		t4 = 2*tau*delta_u * (np.sin(phi)*np.cos(theta) - np.cos(phi)*np.sin(theta)*sin_beta)
		u = np.sqrt(t1+t2+t3+t4)
		parallax  = (u**2+2)/(u*np.sqrt(u**2+4))
		out[i] = - 2.5*np.log10(blend*np.power(10, mag/-2.5) + (1-blend)*np.power(10, mag/-2.5) * parallax)
	return out

@nb.jit
def microlens_simple(t, mag, blend, u0, t0, tE, delta_u, theta):
	u = np.sqrt(u0*u0 + ((t-t0)**2)/tE/tE)
	amp = (u**2+2)/(u*np.sqrt(u**2+4))
	return - 2.5*np.log10(blend*np.power(10, mag/-2.5) + (1-blend)*np.power(10, mag/-2.5) * amp)

import pandas as pd

def distance1(cnopa, cpara):
	return np.max(np.abs(cnopa-cpara))

def distance2(cnopa, cpara):
	return np.abs(cnopa-cpara).sum()/np.sum(19.-cnopa)

@nb.jit(nopython=True)
def dtw_distance(cnopa, cpara):
	dtw = list(np.full(shape=(len(cpara), len(cnopa)), fill_value=np.inf))
	dtw[0][0] = 0.
	for i in range(1, len(cnopa)):
		for j in range(1, len(cpara)):
			cost = (cnopa[i]-cpara[j])**2
			dtw[i][j] = cost #+ np.min([dtw[i][j-1], dtw[i-1][j-1], dtw[i-1][j]])
	print(dtw[-1][-1])
	return dtw[-1][-1]


def fastdtw_distance(cnopa, cpara):
	distance, path = fastdtw.fastdtw(cnopa, cpara, dist=euclidean)
	print(distance)
	# print(path)
	# path = np.array(path)
	# plt.plot(cnopa[path[:,0]])
	# plt.plot(cpara[path[:,1]])
	# plt.plot(cnopa, linestyle=":")
	# plt.plot(cpara, linestyle=":")
	# plt.gca().invert_yaxis()
	# plt.show()
	return distance

def peak_distance(cnopa, cpara, min_prominence=0., base_mag=19.):
	peaks, infos = find_peaks(cpara-base_mag, prominence=min_prominence)
	if len(peaks):
		return len(peaks)#np.max(infos["prominences"])
	else :
		return 0

def scipy_simple_fit_distance(cnopa, cpara, time_range, init_params):
	def fitter_func(params):
		u0, t0, tE = params
		return np.max(np.abs((cpara - microlens_simple(time_range, 19., 0., u0, t0, tE, 0., 0.))))

	res = scipy.optimize.minimize(fitter_func, x0=[init_params['u0'], init_params['t0'], init_params['tE']], method='Nelder-Mead')
	return res.fun

def max_fitter(t, u0, t0, tE, pu0, pt0, ptE, pdu, ptheta):
	return -np.abs((microlens_parallax(t, 19, 0, pu0, pt0, ptE, pdu, ptheta) - microlens_simple(t, 19., 0., u0, t0, tE, 0., 0.)))

def fit_minmax_distance(cnopa, cpara, time_range, init_params):
	def fitter_minmax(u0, t0, tE):
		return - scipy.optimize.differential_evolution(max_fitter, bounds=[(init_params['t0']-400, init_params['t0']+400)], args=(u0, t0, tE, init_params['u0'], init_params['t0'], init_params['tE'], init_params['delta_u'], init_params['theta']), disp=False, popsize=40, mutation=(0.5, 1.0)).fun
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
		distance_args = {'time_range': t, 'init_params':{'u0':params['u0'], 'tE':params['tE'], 't0':params['t0'], 'delta_u':params['delta_u'], 'theta':params['theta']}}
		# cnopa = microlens_simple(t, **params)
		# cpara = microlens_parallax(t, **params)
		# nopa_center = numba_weighted_mean(t, 19 - cnopa)
		# para_center = numba_weighted_mean(t, 19 - cpara)
		# shift = nopa_center - para_center
		st1 = time.time()
		ds.append(distance(microlens_simple(t, **params), microlens_parallax(t, **params), **distance_args))
		print(time.time()-st1)
	return ds


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
		color_val = (pd.cut(dfi[curr_muparameter], bins=rbins).value_counts(sort=False) / len(dfi)).to_numpy()
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
# all_params=[]
# np.random.seed(1234567890)
# all_xvts = np.load('../test/xvt_samples.npy')
# idx = np.arange(0, len(all_xvts)-1)
# np.random.shuffle(idx)
# print(all_xvts[idx[0]])
# all_xvts = all_xvts[idx]
# plt.hist2d(delta_u_from_x(all_xvts[:,0], mass=60.), tE_from_xvt(all_xvts[:,0], all_xvts[:,1], mass=60.), bins=300, range=((0, 0.05), (0, 1000)))
# plt.show()
# all_xvts = all_xvts[:10]
# plt.hist2d(delta_u_from_x(all_xvts[:,0], mass=60.), tE_from_xvt(all_xvts[:,0], all_xvts[:,1], mass=60.), bins=300, range=((0, 0.05), (0, 1000)))
# plt.show()
# for mass in np.geomspace(0.1, 1000, 5):
# 	for g in all_xvts:
# 		all_params.append(generate_parameters(mass=mass, x=g[0], vt=g[1]))
# df = pd.DataFrame.from_records(all_params)
#
#
# st1 = time.time()
# ds = compute_distance(all_params, distance=fit_minmax_distance, time_sampling=1000)
# print(time.time()-st1)
#
# df = df.assign(distance=ds)
# df.to_pickle('temp_fittermax.pkl')

# df = pd.read_pickle('temp_maxdiff.pkl')
# df = pd.read_pickle('temp_peaknb.pkl')
# df = pd.read_pickle('temp_maxpeaksprom.pkl')
# df = pd.read_pickle('temp_meanpeaksprom.pkl')
# df = pd.read_pickle('temp_peaksprom.pkl')
# df = pd.read_pickle('temp_fitter.pkl')
df = pd.read_pickle('temp_fittermax.pkl')
df = df.join(pd.DataFrame(df.pop('distance').to_list())[['fval', 'is_valid']])
df.rename(columns={'fval':'distance'}, inplace=True)
#df = df[df.is_valid]


# cutoff_list = [0.01, 0.05, 0.1, 1]
# for mass in np.geomspace(0.1, 1000, 9):
# 	print(mass)
# 	print(len(df[(df.mass==mass) & (df.distance>1)]) / len(df[df.mass==mass]))
# 	print("------------")
# print(len(df))
# cutoff_list = [1, 2]
cutoff_list = [1, 10, 100]

# parameter_space(df[df['mass']==10.])
# parameter_space(df[df['mass']==100.])
# sns.pairplot(df[df.mass==10.], hue='distance', vars=['u0', 'tE', 'delta_u'])

# fraction(df, cutoff=cutoff_list, bins=np.linspace(0.9, 3.1, 10), binfunc=lambda x: np.power(10, x), show_tot=True)
# fraction(df, curr_muparameter='u0', cutoff=cutoff_list, bins=np.linspace(0, 1, 20), show_tot=False)
# fraction(df, curr_muparameter='tE', cutoff=cutoff_list, bins=np.linspace(1, 5, 20), binfunc=lambda x: np.power(10, x), show_tot=True)
# fraction(df, curr_muparameter='delta_u', cutoff=cutoff_list, bins=np.linspace(0, 0.2, 20), show_tot=True)
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
del p1['is_valid']
t = np.linspace(tmin, tmax, 10000)
cnopa = microlens_simple(t, **p1)
cpara = microlens_parallax(t, **p1)

fig, axs = plt.subplots(ncols=1, nrows=2, sharex='col')
pdif1, = axs[1].plot(t, np.abs((microlens_parallax(t, 19, 0, p1['u0'], p1['t0'], p1['tE'], p1['delta_u'],p1['theta']) - microlens_simple(t, 19., 0., p1['u0'], p1['t0'], p1['tE'], 0., 0.))))
ppar1, = axs[0].plot(t, -(microlens_parallax(t, 19, 0, p1['u0'], p1['t0'], p1['tE'], p1['delta_u'], p1['theta'])))
pnop1, = axs[0].plot(t, -(microlens_simple(t, 19, 0, p1['u0'], p1['t0'], p1['tE'], p1['delta_u'], p1['theta'])))
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
	explored_parameters.append([u0, t0, tE, r])


def max_fitter(t, u0, t0, tE, pu0, pt0, ptE, pdu, ptheta):
	return -np.abs((microlens_parallax(t, 19, 0, pu0, pt0, ptE, pdu, ptheta) - microlens_simple(t, 19., 0., u0, t0, tE, 0., 0.)))

def fitter_minmax(u0, t0, tE):
	res = scipy.optimize.differential_evolution(max_fitter, bounds=[(p1['t0']-400, p1['t0']+400)], args=(u0, t0, tE, p1['u0'], p1['t0'], p1['tE'], p1['delta_u'], p1['theta']), disp=False, popsize=40, mutation=(0.5, 1.0), strategy='randtobest1bin')
	# tm = Minuit(max_fitter, t=t0, error_t=100, limit_t=(tmin, tmax) ,errordef=1, print_level=0)
	# tm.migrad()
	if isinstance(res.fun, np.ndarray):
		r = res.fun[0]
	else:
		r = res.fun
	update_plot(u0, t0, tE, r)
	return -r

m = Minuit(fitter_minmax,
		   u0=p1['u0'],
		   t0=p1['t0'],
		   tE=p1['tE'],
		   error_u0=0.1,
		   error_t0=100,
		   error_tE=100,
		   limit_u0=(0, 2),
		   limit_tE=(p1['tE']*(1-np.sign(p1['tE'])*0.5), p1['tE']*(1+np.sign(p1['tE'])*0.5)),
		   limit_t0=(p1['t0']-abs(p1['tE']), p1['t0']+abs(p1['tE'])),
		   errordef=1,
		   print_level=1
		   )

def onclick(event):
	m.migrad()
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
print(m.values)
cnopa = microlens_simple(t, 19., 0., m.values['u0'], m.values['t0'], m.values['tE'], 0., 0.)

# explored_parameters = np.array(explored_parameters)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# print(explored_parameters.shape)
# ax.scatter(explored_parameters[:,0], explored_parameters[:,1], explored_parameters[:,3])
# plt.show()

# print(np.max(np.abs(cpara-cnopa)))
# m.draw_profile('tE')
# plt.show()
# m.draw_profile('u0')
# plt.show()
# xbins, ybins, values = m.contour('tE', 'u0', bound=[[-5, 100], [0, 1]])
# plt.contour(values, extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]])
# plt.show()

# def fitter_func(params):
# 	u0, t0, tE = params
# 	res = scipy.optimize.differential_evolution(
# 		lambda t: -np.abs((microlens_parallax(t, 19, 0, p1['u0'], p1['t0'], p1['tE'], p1['delta_u'], p1['theta']) - microlens_simple(t, 19., 0., u0, t0, tE, 0., 0.))),
# 		bounds=[(tmin, tmax)], disp=False)
# 	plt.plot(t, -np.abs((microlens_parallax(t, 19, 0, p1['u0'], p1['t0'], p1['tE'], p1['delta_u'], p1['theta']) - microlens_simple(t, 19., 0., u0, t0, tE, 0., 0.))))
# 	plt.plot(t, -(microlens_parallax(t, 19, 0, p1['u0'], p1['t0'], p1['tE'], p1['delta_u'], p1['theta'])))
# 	plt.plot(t, -(microlens_simple(t, 19, 0, u0, t0, tE, p1['delta_u'], p1['theta'])))
# 	plt.axvline(res.x)
# 	plt.xlim(51200, 51600)
# 	plt.show()
# 	return - res.fun
#
# res = scipy.optimize.minimize(fitter_func, x0=[p1['u0'], p1['t0'], p1['tE']], method='Nelder-Mead')
# print(res)
# cnopa = microlens_simple(t, 19., 0., res.x[0], res.x[1], res.x[2], 0., 0.)


# nopa_center = numba_weighted_mean(t, 19 - cnopa)
# para_center = numba_weighted_mean(t, 19 - cpara)
# shift = nopa_center - para_center
# print(shift)
# cpara = microlens_parallax(t+shift, **p1)

fig, axs = plt.subplots(nrows=2, ncols=1, sharex='col')
axs[0].plot(t, cnopa)
axs[0].plot(t, cpara)
axs[0].invert_yaxis()
axs[1].plot(t, np.abs(cnopa-cpara))
axs[1].invert_yaxis()
plt.figure()
nb_bins=100
range_tE = (0, 1000)
range_delta_u = (0, 0.05)
plt.hist2d(df['tE'], df['delta_u'], bins=nb_bins, range=(range_tE, range_delta_u))
plt.scatter(p1['tE'], p1['delta_u'], marker='x', s=100, color='black')
plt.show()