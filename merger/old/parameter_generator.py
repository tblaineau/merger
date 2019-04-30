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
	return p_xvt(x, vt, mass)*rho_halo(x)/mass*x*x*f_vt(vt)

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


def pdf_tEdu(t_E, delta_u, mass):
	x = x_from_delta_u(delta_u, mass)
	vt = v_T_from_tEdu(delta_u, t_E, mass)
	return p_xvt(x, vt, mass)*rho_halo(x)/mass*x*x*f_vt(vt)*np.abs(jacobian(delta_u, t_E, mass))

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
		samples.append(current_x)
	print(accepted, accepted/nb_samples)
	return np.array(samples)

def generate_parameters(mass, seed=None, blending=False, parallax=False, s=None):
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
	if not isinstance(s, np.ndarray):
		s = np.load('xvt_samples.npy')
	x , vt = s[np.random.randint(0, s.shape[0])]
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
	return np.max(np.abs(cnopa-cpara))/np.max((19.-cnopa))

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
		return len(peaks)
	else :
		return 0

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
	for params in params_set:
		del params['mass']
		params['mag']=19.
		params['blend']=0.
		# cnopa = microlens_simple(t, **params)
		# cpara = microlens_parallax(t, **params)
		# nopa_center = numba_weighted_mean(t, 19 - cnopa)
		# para_center = numba_weighted_mean(t, 19 - cpara)
		# shift = nopa_center - para_center
		ds.append(distance(microlens_simple(t, **params), microlens_parallax(t, **params)))
	return ds

# s = np.load('xvt_samples.npy')
# s = s[(s[:,0]>0) & (s[:,1]>0)]
# all_params=[]
# np.random.seed(1234567890)
# for mass in np.random.uniform(10, 1000, size=500000):
# 	all_params.append(generate_parameters(mass=mass, s=s))
# df = pd.DataFrame.from_records(all_params)
#
#
# st1 = time.time()
# ds = compute_distance(all_params, distance=peak_distance)
# print(time.time()-st1)
#
# df = df.assign(distance=ds)
# df.to_pickle('temp_peaknb.pkl')

# df = pd.read_pickle('temp_peaknb.pkl')
df = pd.read_pickle('temp_maxpeaksprom.pkl')
# df = pd.read_pickle('temp_meanpeaksprom.pkl')
# df = pd.read_pickle('temp_peaksprom.pkl')


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
		sns.scatterplot(tdf['u0'], tdf['delta_u'], size=tdf['mass'], hue=tdf['tE'])
		plt.plot([[0,0], [1,1]], ls="--", c=".3")
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
		ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, ticknb: "{:.1f}".format(binfunc(value))))
	plt.legend()
	ax.legend()
	plt.show()


fraction(df, cutoff=[0.01, 0.05, 0.127], bins=np.linspace(1, 3, 20), binfunc=lambda x: np.power(10, x), show_tot=False)
fraction(df, curr_muparameter='u0', cutoff=[0.01, 0.05, 0.127], bins=np.linspace(0, 1, 20), show_tot=False)
fraction(df, curr_muparameter='tE', cutoff=[0.01, 0.05, 0.127], bins=np.linspace(2, 5, 20), binfunc=lambda x: np.power(10, x), show_tot=True)
cuttoffs_scatter_plots(df, cutoffs=[0.01, 0.05, 0.127])
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
t = np.linspace(tmin, tmax, 1000)
cnopa = microlens_simple(t, **p1)
cpara = microlens_parallax(t, **p1)

# nopa_center = numba_weighted_mean(t, 19 - cnopa)
# para_center = numba_weighted_mean(t, 19 - cpara)
# shift = nopa_center - para_center
# print(shift)
# cpara = microlens_parallax(t+shift, **p1)

plt.subplot(211)
plt.plot(t, cnopa)
plt.plot(t, cpara)
plt.gca().invert_yaxis()
plt.subplot(212)
plt.plot(t, cnopa-cpara)
plt.gca().invert_yaxis()
plt.show()