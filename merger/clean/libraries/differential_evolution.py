import numpy as np
import numba as nb
from merger.clean.libraries.merger_library import COLOR_FILTERS
from iminuit import Minuit
import logging
import pandas as pd
from sklearn.utils.random import sample_without_replacement
from scipy.stats import kstest, norm, ks_2samp

fastmath = False

# We define parallax parameters.
PERIOD_EARTH = 365.2422
alphaS = 80.8941667*np.pi/180.
deltaS = -69.7561111*np.pi/180.
epsilon = (90. - 66.56070833)*np.pi/180.		# source in LMC
t_origin = 51442 								# (21 septembre 1999) #58747 #(21 septembre 2019)

sin_beta = np.cos(epsilon)*np.sin(deltaS) - np.sin(epsilon)*np.cos(deltaS)*np.sin(alphaS)
beta = np.arcsin(sin_beta) 						# ok because beta is in -pi/2; pi/2
if abs(beta)==np.pi/2:
	lambda_star = 0
else:
	lambda_star = np.sign((np.sin(epsilon)*np.sin(deltaS)+np.cos(epsilon)*np.sin(alphaS)*np.cos(deltaS))/np.cos(beta)) * np.arccos(np.cos(deltaS)*np.cos(alphaS)/np.cos(beta))


@nb.njit(fastmath=fastmath)
def microlens_parallax(t, mag, blend, u0, t0, tE, delta_u, theta):
	tau = (t-t0)/tE
	phi = 2*np.pi * (t-t_origin)/PERIOD_EARTH - lambda_star
	t1 = u0**2 + tau**2
	t2 = delta_u**2 * (np.sin(phi)**2 + np.cos(phi)**2*sin_beta**2)
	t3 = -2*delta_u*u0 * (np.sin(phi)*np.sin(theta) + np.cos(phi)*np.cos(theta)*sin_beta)
	t4 = 2*tau*delta_u * (np.sin(phi)*np.cos(theta) - np.cos(phi)*np.sin(theta)*sin_beta)
	u = np.sqrt(t1+t2+t3+t4)
	parallax = (u**2+2)/(u*np.sqrt(u**2+4))
	return - 2.5*np.log10(blend + (1-blend)* parallax) + mag


@nb.njit(fastmath=fastmath)
def microlens_simple(t, mag, blend, u0, t0, tE, delta_u=0, theta=0):
	u = np.sqrt(u0*u0 + ((t-t0)**2)/tE/tE)
	amp = (u**2+2)/(u*np.sqrt(u**2+4))
	return - 2.5*np.log10(blend + (1-blend)* amp) + mag


@nb.njit(fastmath=fastmath)
def microlens_simple_flux(t, flux, blend, u0, t0, tE, delta_u=0, theta=0):
	u = np.sqrt(u0*u0 + ((t-t0)**2)/tE/tE)
	amp = (u**2+2)/(u*np.sqrt(u**2+4))
	return flux * ((1 - blend) * amp + blend)


@nb.njit(fastmath=fastmath)
def inner_loop(x, a, b, c, dim, recombination, bounds):
	y = []
	for i in range(dim):
		ri = np.random.uniform(0, 1)
		temp = a[i] + np.random.uniform(0.5, 1.) * (b[i] - c[i])
		if ri < recombination:
			if bounds[i][0] < temp and temp < bounds[i][1]:
				y.append(temp)
			else:
				y.append(np.random.uniform(bounds[i][0], bounds[i][1]))
		else:
			y.append(x[i])
	return y


@nb.njit(fastmath=fastmath)
def main_loop(func, times, data, errors, dim, recombination, init_pop, pop, all_values, best_idx, bounds):
	for i in range(pop):
		x = init_pop[i]
		idx = np.random.choice(pop, 2, replace=False)
		y = inner_loop(x, init_pop[best_idx], init_pop[idx[0]], init_pop[idx[1]], dim, recombination, bounds)
		cval = func(y, times, data, errors)
		if cval <= func(x, times, data, errors):
			all_values[i] = func(y, times, data, errors)
			init_pop[i] = y
		if cval < all_values[best_idx]:
			best_idx = i
	return best_idx


@nb.njit(fastmath=fastmath)
def diff_ev_init_pop(func, times, data, errors, bounds, init_pop, recombination=0.7, tol=0.01):
	"""
	Compute minimum func value using differential evolution algorithm with input population

	Parameters
	----------
	func : function
		function to minimize, of format func(parameters, time, data, errors)
	times : sequence
		time values
	data : sequence
	errors : sequence
	bounds : np.array
		Limits of the parameter value to explore
		len(bounds) should be the number of parameters to func
	init_pop : np.array
		initial population
	recombination : float
		Recombination factor, fraction of non mutated specimen to next generation
		Should be in [0, 1]
	tol : float
		Tolerance factor, used for stopping condition

	Returns
	-------
	tuple(float, list, int)
		Returns minimum function value, corresponding parameters and number of loops

	"""
	dim = len(bounds)
	pop = len(init_pop)

	all_values = []
	for i in range(pop):
		all_values.append(func(init_pop[i], times, data, errors))
	all_values = np.array(all_values)
	best_idx = all_values.argmin()
	count = 0
	# loop
	while count < 1000:
		best_idx = main_loop(func, times, data, errors, dim, recombination, init_pop, pop, all_values, best_idx, bounds)
		count += 1

		if np.std(all_values) <= np.abs(np.mean(all_values)) * tol:
			break
	# rd = np.mean(all_values) - min_val
	# rd = rd**2/(min_val**2 + eps)
	# if rd<eps and count>20:
	#    break
	return all_values[best_idx], init_pop[best_idx], count


@nb.njit(fastmath=fastmath)
def diff_ev(func, times, data, errors, bounds, pop, recombination=0.7, tol=0.01):
	"""
	Compute minimum func value using differential evolution algorithm.
	6 times faster than scipy.optimize.differential_evolution, (~45ms vs ~275ms)

	Parameters
	----------
	func : function
		function to minimize, of format func(parameters, time, data, errors)
	times : sequence
		time values
	data : sequence
	errors : sequence
	bounds : np.array
		Limits of the parameter value to explore
		len(bounds) should be the number of parameters to func
	pop : int
		Number of specimen to evolve
	recombination : float
		Recombination factor, fraction of non mutated specimen to next generation
		Should be in [0, 1]
	tol : float
		Tolerance factor, used for stopping condition

	Returns
	-------
	tuple(float, list, int)
		Returns minimum function value, corresponding parameters and number of loops

	"""
	dim = len(bounds)
	# Initialize
	init_pop = []
	for i in range(pop):
		p = []
		for j in range(dim):
			p.append(np.random.uniform(bounds[j, 0], bounds[j, 1]))
		init_pop.append(p)

	all_values = []
	for i in range(pop):
		all_values.append(func(init_pop[i], times, data, errors))
	all_values = np.array(all_values)
	best_idx = all_values.argmin()
	count = 0
	# loop
	while count < 1000:
		best_idx = main_loop(func, times, data, errors, dim, recombination, init_pop, pop, all_values, best_idx, bounds)
		count += 1

		if np.std(all_values) <= np.abs(np.mean(all_values)) * tol:
			break
	# rd = np.mean(all_values) - min_val
	# rd = rd**2/(min_val**2 + eps)
	# if rd<eps and count>20:
	#    break
	return all_values[best_idx], init_pop[best_idx], count


@nb.njit(fastmath=fastmath)
def to_minimize_simple(params, t, tx, errx):
	mag, u0, t0, tE = params
	s=0
	for i in range(len(t)):
		s += (tx[i] - microlens_simple(t[i], mag, 0, u0, t0, tE))**2/errx[i]**2
	return s

@nb.njit(fastmath=fastmath)
def to_minimize_parallax(params, t, tx, errx):
	mag, u0, t0, tE, blend, delta_u, theta = params
	s=0
	for i in range(len(t)):
		s+= (tx[i] - microlens_parallax(t[i], mag, blend, u0, t0, tE, delta_u, theta))**2/errx[i]**2
	return s


@nb.njit
def nb_truncated_intrinsic_dispersion(time, mag, err, fraction=0.05):
	s0 = []
	for i in range(0, len(time)-2):
		ri = (time[i+1]-time[i])/(time[i+2]-time[i])
		sigmaisq = err[i+1]**2 + (1-ri)**2 * err[i]**2 + ri**2 * err[i+2]**2
		s0.append(((mag[i+1] - mag[i] - ri*(mag[i+2]-mag[i]))**2/sigmaisq))
	maxind = int(len(s0) * fraction)
	s0 = np.array(s0)
	s0 = s0[s0.argsort()[:len(s0)-maxind]].sum()
	return np.sqrt(s0/(len(time)-2-maxind))


@nb.njit
def nb_truncated_sigint(time, mag, fraction=0.05):
	s0 = []
	for i in range(0, len(time)-2):
		ri = (time[i+1]-time[i])/(time[i+2]-time[i])
		s0.append((mag[i+1] - mag[i] - ri*(mag[i+2]-mag[i]))**2/(2*ri**2-2*ri+2))
	maxind = int(len(s0) * fraction)
	s0 = np.array(s0)
	s0 = s0[s0.argsort()[:len(s0)-maxind]].sum()
	return np.sqrt(s0/(len(time)-2-maxind))


@nb.njit
def to_minimize_simple_nd(params, t, tx, errx):
	"params : mags_1...mags_n, u0, t0, tE"
	u0 = params[-3]
	t0 = params[-2]
	tE = np.power(10, params[-1])
	s=0
	for j in range(len(params)-3):
		mag = params[j]
		#s+=loop4d_simple(mag, u0, t0, tE, t[j], tx[j], errx[j])
		for i in range(len(t[j])):
			s += (tx[j][i] - microlens_simple(t[j][i], mag, 0, u0, t0, tE))**2/errx[j][i]**2
	return s


@nb.njit
def to_minimize_simple_flux(params, t, tx, errx):
	"params : mags_1...mags_n, u0, t0, tE"
	u0 = params[-3]
	t0 = params[-2]
	tE = np.power(10, params[-1])
	s=0
	for j in range(len(params)-3):
		flux = params[j]
		#s+=loop4d_simple(mag, u0, t0, tE, t[j], tx[j], errx[j])
		for i in range(len(t[j])):
			s += (tx[j][i] - microlens_simple_flux(t[j][i], flux, 0, u0, t0, tE))**2 / errx[j][i]**2
	return s

@nb.njit
def to_minimize_blended_nd(params, t, tx, errx):
	"params : mags_1...mags_n, u0, t0, tE"
	u0 = params[-3]
	t0 = params[-2]
	tE = np.power(10, params[-1])
	s=0
	npms = int((len(params)-3)/2.)
	for j in range(npms):
		mag = params[j]
		blend = params[npms+j]
		#s+=loop4d_simple(mag, u0, t0, tE, t[j], tx[j], errx[j])
		for i in range(len(t[j])):
			s += (tx[j][i] - microlens_simple(t[j][i], mag, blend, u0, t0, tE))**2/errx[j][i]**2
	return s



@nb.njit
def to_minimize_parallax_nd(params, t, tx, errx):
	"params : mags_1...mags_n, u0, t0, tE, du, theta"
	u0 = params[-5]
	t0 = params[-4]
	tE = params[-3]
	delta_u = params[-2]
	theta = params[-1]
	s=0
	for j in range(len(params)-5):
		mag = params[j]
		#s+=loop4d_parallax(mag, u0, t0, tE, delta_u, theta, t[j], tx[j], errx[j])
		for i in range(len(t[j])):
			s += (tx[j][i] - microlens_parallax(t[j][i], mag, 0, u0, t0, tE, delta_u, theta))**2/errx[j][i]**2
	return s


def diff_ev_lhs(func, times, data, errors, bounds, pop, recombination=0.7, tol=0.01):
	"""Compute minimum func value using differential evolution algorithm with input population generated using LHS"""
	init_pop = latin_hypercube_sampling(bounds, pop)
	return diff_ev_init_pop(func, times, data, errors, bounds, init_pop, recombination, tol)


def latin_hypercube_sampling(bounds, pop):
	"""Latin Hypercube sampling to generate more uniformly distributed differential evolution initial parameters values.

	Parameters
	----------
	bounds : np.array
		Bounds to generate parameters within, should be of shape (nb of parameters, 2)
	pop : int
		Number of sets of inital parameters to generate
	"""
	ranges = np.linspace(bounds[:, 0], bounds[:, 1], pop + 1).T
	ranges = np.array([ranges[:,:-1], ranges[:,1:]]).T
	cs = np.random.uniform(low=ranges[:,:,0], high=ranges[:,:,1])
	a = np.random.sample(cs.T.shape).argsort()
	return np.array([cs[a[i], i] for i in range(len(a))]).T


GLOBAL_COUNTER = 0

def fit_ml_de_flux(subdf, do_cut5=False, hesse=False, minos=False):
	"""Fit on one star

	Color filter names must be stocked in a COLOR_FILTERS dictionnary
	for example : COLOR_FILTERS = {"r":{"mag":"mag_r", "err":"magerr_r"},
								   "g":{"mag":"mag_g", "err":"magerr_g"}}

	Parameters
	----------
	subdf : pd.DataFrame
		Lightcurve data. Should have magnitudes stocked in "mag_*color*"  columns, magnitude errors in "magerr_*color*",
		for each *color* name and timestamps in "time" column

	do_cut5 : bool
		If True, clean aberrant points using distance from median of 5 points (default: {False})

	hesse : bool
		Use HESSE to compute paraoblic errors (not really valid as the chi2 profile is not parabolic, for tE)

	minos : bool
		Use MINOS to compute asymetric errors (slow the function, ~x1.5)

	Returns
	-------
	pd.Series
		Contains parameters for the microlensing and flat curve fits, their chi2, informations on the fitter (fmin) and dof :

		mulens Fit results parameters : mag_1, ... mag_n, u0, t0, tE
		mulens Minuit fit output informations
		mulens Fit final Chi^2
		flat Fit results parameters : mag_1, ... mag_n
		flat Minuit fit output informations
		flat Fit final Chi^2
		Number of points used in each color filter
		Individual final mulens fit chi^2 value for each filter
		Individual final flat fit chi^2 value for each filter
		Intrinsic dispersion for each color filter
	"""

	# print(subdf.name)

	mask = dict()
	errs = dict()
	flux = dict()
	cut5 = dict()
	time = dict()

	min_err = 0.0
	remove_extremities = True
	tolerance_ratio = 0.9
	p = True
	ufilters = []

	for key in COLOR_FILTERS.keys():
		mask[key] = (subdf[COLOR_FILTERS[key]["flux"]].notnull() &
					 subdf[COLOR_FILTERS[key]["fluxerr"]].notnull() &
					 subdf[COLOR_FILTERS[key]["err"]].between(min_err, 9.999, inclusive=False))  # No nan and limits on errors

		if mask[key].sum()>4:		#Check if there are more than 3 valid points in the current color
			ufilters.append(key)
			flux[key] = subdf[mask[key]][COLOR_FILTERS[key]["flux"]]  # mags
			errs[key] = subdf[mask[key]][COLOR_FILTERS[key]["fluxerr"]]  # errs
			cut5[key] = np.abs((flux[key].rolling(5, center=True).median() - flux[key][2:-2])) / errs[key][2:-2] < 5

			if not remove_extremities:
				cut5[key][:2] = True
				cut5[key][-2:] = True

			p *= cut5[key].sum() / len(cut5[key]) < tolerance_ratio

	if do_cut5 and not p:
		for key in ufilters:
			time[key] = subdf[mask[key]][cut5[key]].time.values
			errs[key] = errs[key][cut5[key]].values
			flux[key] = flux[key][cut5[key]].values
	else:
		for key in ufilters:
			time[key] = subdf[mask[key]].time.values
			errs[key] = errs[key].values
			flux[key] = flux[key].values

	# Normalize errors
	intrinsic_dispersion = dict()
	median_errors = dict()
	scount = dict()
	for key in COLOR_FILTERS.keys():
		intrinsic_dispersion[key] = np.nan
		median_errors[key] = np.nan
		scount[key] = 0
	for key in ufilters:
		median_errors[key] = np.median(errs[key])
		if len(flux[key]) <= 3:
			intrinsic_dispersion[key] = 1.
		else:
			tmsk = errs[key] < 0.6 * flux[key]
			scount[key] = tmsk.sum()
			try :
				intrinsic_dispersion[key] = nb_truncated_intrinsic_dispersion(time[key][tmsk], flux[key][tmsk], errs[key][tmsk], fraction=0.01)
			except ZeroDivisionError:
				print("invalid intr_dispersion for "+key+" : ZeroDivisionError.")
			if intrinsic_dispersion[key] > 0:
				errs[key] = errs[key] * intrinsic_dispersion[key]
			else:
				print(f"null intrinsic dispersion for {subdf.name}, key : {key}, Length : {mask[key].sum()}")

	# if magRE.size==0 or magBE.size==0 or magRM.size==0 or magBM.size==0:
	# 	return pd.Series(None)

	# flat fit
	def least_squares_flat(x):
		s = 0
		for idx, key in enumerate(ufilters):
			s += np.sum(((flux[key] - x[idx]) / errs[key]) ** 2)
		return s

	start = [np.median(flux[key]) for key in ufilters]
	error = [1. for _ in ufilters]
	name = ["f_fluxStar_" + key for key in ufilters]
	m_flat = Minuit.from_array_func(least_squares_flat,
									start=start,
									error=error,
									name=name,
									errordef=1,
									print_level=0)
	m_flat.migrad()
	global GLOBAL_COUNTER
	GLOBAL_COUNTER += 1
	# print(str(GLOBAL_COUNTER) + " : " + subdf.name)
	flat_params = m_flat.values

	# init for output
	flat_keys = ["f_fluxStar_" + key for key in COLOR_FILTERS.keys()]
	flat_values = []
	for key in COLOR_FILTERS.keys():
		if key in ufilters:
			flat_values.append(m_flat.values["f_fluxStar_" + key])
		else:
			flat_values.append(np.nan)
	flat_fmin = m_flat.get_fmin()
	flat_fval = m_flat.fval

	alltimes = np.concatenate(list(time.values()))
	tmin = alltimes.min()
	tmax = alltimes.max()
	try:
		bounds_simple = np.array([[0, np.max(flux[key])] for key in ufilters] + [[0, 3], [tmin-600, tmax+600], [0, 3.7]])
	except ValueError as e:
		print(p)
		print(cut5)
		print(ufilters)
		print(str(e))
	try:
		fval, pms, nbloops = diff_ev_lhs(to_minimize_simple_flux, list(time.values()), list(flux.values()),
									 list(errs.values()), bounds=bounds_simple, pop=70, recombination=0.3)
	except ZeroDivisionError:
		logging.error("Divison by zero in diffev. Current star: "+str(subdf.name))

	names = ["u0", "t0", "tE"] + ["fluxStar_" + key for key in COLOR_FILTERS.keys()]
	micro_keys = names

	pms = list(pms)

	def least_squares_microlens(x):
		lsq = 0
		for idx, key in enumerate(ufilters):
			lsq += np.sum(((flux[key] - microlens_simple_flux(time[key], x[idx + 3], 0., x[0], x[1], x[2])) / errs[key]) ** 2)
		return lsq

	start = pms[-3:-1] + [np.power(10, pms[-1])] + pms[:-3]
	names = ["u0", "t0", "tE"] + ["fluxStar_" + key for key in ufilters]
	errors = [0.1, 100, 10] + [0.1*pms[i] for i in range(len(ufilters))]
	limits = [(0, 3), (tmin - 1000, tmax + 1000), (1, 5000)] + [(None, None) for _ in ufilters]
	m_micro = Minuit.from_array_func(least_squares_microlens,
									 start=start,
									 error=errors,
									 limit=limits,
									 name=names,
									 errordef=1,
									 print_level=0)

	m_micro.migrad()
	micro_params = m_micro.values

	micro_errors = []
	micro_error_labels = []

	if hesse:
		micro_error_labels = ["error_" + name for name in micro_keys]
		micro_errors = [np.nan]*len(micro_keys)
		m_micro.hesse()
		micro_errors = [m_micro.errors["u0"], m_micro.errors["t0"], m_micro.errors["tE"]]
		for key in COLOR_FILTERS.keys():
			if key in ufilters:
				micro_errors.append(m_micro.errors["fluxStar_" + key])
			else:
				micro_errors.append(np.nan)
	elif minos:
		for name in micro_keys:
			micro_error_labels+=["lower_error_"+name, "upper_error_"+name, "valid_lower_error_"+name, "valid_upper_error_"+name]
		micro_errors = [np.nan]*(3+len(COLOR_FILTERS))*4
		merrors=False
		try:
			merrors = m_micro.minos()
		except RuntimeError:
			print("Migrad did not converge properly on star " + str(subdf.name))
		if merrors:
			micro_errors = [merrors["u0"].lower, merrors["u0"].upper, merrors["u0"].lower_valid, merrors["u0"].upper_valid,
							merrors["t0"].lower, merrors["t0"].upper, merrors["t0"].lower_valid,
							merrors["t0"].upper_valid,
							merrors["tE"].lower, merrors["tE"].upper, merrors["tE"].lower_valid,
							merrors["tE"].upper_valid]
			for key in COLOR_FILTERS.keys():
				if key in ufilters:
					micro_errors += [merrors["fluxStar_" + key].lower, merrors["fluxStar_" + key].upper,
										 merrors["fluxStar_" + key].lower_valid, merrors["fluxStar_" + key].upper_valid]
				else:
					micro_errors+=[np.nan]*4

	lsqs = []
	micro_values = [micro_params['u0'], micro_params['t0'], micro_params['tE']]
	micro_distributions = dict()
	for key in COLOR_FILTERS.keys():
		if key in ufilters:
			micro_distributions[key] = (flux[key] - microlens_simple_flux(time[key], micro_params["fluxStar_" + key], 0, micro_params['u0'], micro_params['t0'], micro_params['tE'])) / errs[key]
			lsqs.append(np.sum(micro_distributions[key]**2))
			micro_values.append(m_micro.values["fluxStar_" + key])
		else:
			lsqs.append(np.nan)
			micro_values.append(np.nan)
	micro_fmin = m_micro.get_fmin()
	micro_fval = m_micro.fval

	counts = []
	flat_chi2s = []
	flat_distributions = dict()
	for key in COLOR_FILTERS.keys():
		if key in ufilters:
			counts.append((~np.isnan(flux[key])).sum())
			flat_distributions[key] = (flux[key] - m_flat.values["f_fluxStar_"+key]) / errs[key]
			flat_chi2s.append(np.sum(flat_distributions[key]**2))
		else:
			counts.append(0)
			flat_chi2s.append(np.nan)

	# Compute K-S Test
	micro_ks_colors = dict()
	flat_ks_colors = dict()
	f2m_ks_colors = dict()
	for key in COLOR_FILTERS.keys():
		if key in ufilters:
			micro_ks_colors[key] = kstest(micro_distributions[key], norm.cdf).pvalue
			flat_ks_colors[key] = kstest(flat_distributions[key], norm.cdf).pvalue
			f2m_ks_colors[key] = ks_2samp(micro_distributions[key], flat_distributions[key]).pvalue
		else:
			micro_ks_colors[key] = np.nan
			flat_ks_colors[key] = np.nan
			f2m_ks_colors[key] = np.nan
	micro_distribution = np.concatenate(list(micro_distributions.values()))
	flat_distribution =  np.concatenate(list(flat_distributions.values()))

	micro_ks = kstest(micro_distribution, norm.cdf).pvalue
	flat_ks = kstest(flat_distribution, norm.cdf).pvalue
	f2m_ks = ks_2samp(micro_distribution, flat_distribution).pvalue

	return pd.Series(
		micro_values + [micro_fmin, micro_fval] + micro_errors
		+ flat_values + [flat_fmin, flat_fval]
		+ counts
		+ list(scount.values())
		+ lsqs
		+ flat_chi2s
		+ list(median_errors.values())
		+ list(intrinsic_dispersion.values())
		+ [tmin, tmax]
		+ [micro_ks, flat_ks, f2m_ks]
		+ list(micro_ks_colors.values()) + list(flat_ks_colors.values()) + list(f2m_ks_colors.values())
		,

		index = micro_keys + ['micro_fmin', 'micro_fval'] + micro_error_labels
			  + flat_keys + ['flat_fmin', 'flat_fval']
			  + ["counts_" + key for key in COLOR_FILTERS.keys()]  # ["counts_RE", "counts_BE", "counts_RM", "counts_BM"]
			  + ["scounts_"+key for key in COLOR_FILTERS.keys()]
			  + ["micro_chi2_" + key for key in COLOR_FILTERS.keys()]  # ['micro_chi2_RE', 'micro_chi2_BE', 'micro_chi2_RM', 'micro_chi2_BM']
			  + ["flat_chi2_" + key for key in COLOR_FILTERS.keys()]  # ['flat_chi2_RE', 'flat_chi2_RM', 'flat_chi2_BE', 'flat_chi2_BM']
			  + ["magerr_" + key + "_median" for key, cf in COLOR_FILTERS.items()]  # ['errRE_median', 'errBE_median', 'errRM_median', 'errBM_median']
			  + ["intr_disp_" + key for key in intrinsic_dispersion.keys()]
			  + ["tmin", "tmax"]
			  + ["micro_ks", "flat_ks", "f2m_ks"]
			  + ["micro_ks_"+key for key in COLOR_FILTERS.keys()]
			  + ["flat_ks_"+key for key in COLOR_FILTERS.keys()]
			  + ["f2m_ks_"+key for key in COLOR_FILTERS.keys()]
	)


def fit_ml_de_simple(subdf, do_cut5=False, hesse=False, minos=False):
	"""Fit on one star

	Color filter names must be stocked in a COLOR_FILTERS dictionnary
	for example : COLOR_FILTERS = {"r":{"mag":"mag_r", "err":"magerr_r"},
								   "g":{"mag":"mag_g", "err":"magerr_g"}}

	Parameters
	----------
	subdf : pd.DataFrame
		Lightcurve data. Should have magnitudes stocked in "mag_*color*"  columns, magnitude errors in "magerr_*color*",
		for each *color* name and timestamps in "time" column

	do_cut5 : bool
		If True, clean aberrant points using distance from median of 5 points (default: {False})

	hesse : bool
		Use HESSE to compute paraoblic errors (not really valid as the chi2 profile is not parabolic, for tE)

	minos : bool
		Use MINOS to compute asymetric errors (slow the function, ~x1.5)

	Returns
	-------
	pd.Series
		Contains parameters for the microlensing and flat curve fits, their chi2, informations on the fitter (fmin) and dof :

		mulens Fit results parameters : mag_1, ... mag_n, u0, t0, tE
		mulens Minuit fit output informations
		mulens Fit final Chi^2
		flat Fit results parameters : mag_1, ... mag_n
		flat Minuit fit output informations
		flat Fit final Chi^2
		Number of points used in each color filter
		Individual final mulens fit chi^2 value for each filter
		Individual final flat fit chi^2 value for each filter
		Intrinsic dispersion for each color filter
	"""

	# print(subdf.name)

	mask = dict()
	errs = dict()
	mags = dict()
	cut5 = dict()
	time = dict()

	min_err = 0.0
	remove_extremities = True
	tolerance_ratio = 0.9
	p = True
	ufilters = []

	for key in COLOR_FILTERS.keys():
		mask[key] = subdf[COLOR_FILTERS[key]["mag"]].notnull() & subdf[COLOR_FILTERS[key]["err"]].notnull() & subdf[COLOR_FILTERS[key]["err"]].between(
			min_err, 9.999, inclusive=False)  # No nan and limits on errors

		if mask[key].sum()>2:		#Check if there are more than 3 valid points in the current color
			ufilters.append(key)
			mags[key] = subdf[mask[key]][COLOR_FILTERS[key]["mag"]]  # mags
			errs[key] = subdf[mask[key]][COLOR_FILTERS[key]["err"]]  # errs
			cut5[key] = np.abs((mags[key].rolling(5, center=True).median() - mags[key][2:-2])) / errs[key][2:-2] < 5

			if not remove_extremities:
				cut5[key][:2] = True
				cut5[key][-2:] = True

			p *= cut5[key].sum() / len(cut5[key]) < tolerance_ratio

	if do_cut5 and not p:
		for key in ufilters:
			time[key] = subdf[mask[key]][cut5[key]].time.to_numpy()
			errs[key] = errs[key][cut5[key]].to_numpy()
			mags[key] = mags[key][cut5[key]].to_numpy()
	else:
		for key in ufilters:
			time[key] = subdf[mask[key]].time.to_numpy()
			errs[key] = errs[key].to_numpy()
			mags[key] = mags[key].to_numpy()

	# Normalize errors
	intrinsic_dispersion = dict()
	median_errors = dict()
	scount = dict()
	for key in COLOR_FILTERS.keys():
		intrinsic_dispersion[key] = np.nan
		median_errors[key] = np.nan
		scount[key] = 0
	for key in ufilters:
		median_errors[key] = np.median(errs[key])
		if len(mags[key]) <= 3:
			intrinsic_dispersion[key] = 1.
		else:
			tmsk = errs[key]<0.6
			scount[key] = tmsk.sum()
			try :
				intrinsic_dispersion[key] = nb_truncated_intrinsic_dispersion(time[key][tmsk], mags[key][tmsk], errs[key][tmsk], fraction=0.0)
			except ZeroDivisionError:
				print("invalid intr_dispersion")
			if intrinsic_dispersion[key] > 0:
				errs[key] = errs[key] * intrinsic_dispersion[key]
			else:
				print(f"null intrinsic dispersion for {subdf.name}")

	# if magRE.size==0 or magBE.size==0 or magRM.size==0 or magBM.size==0:
	# 	return pd.Series(None)

	# flat fit
	def least_squares_flat(x):
		s = 0
		for idx, key in enumerate(ufilters):
			s += np.sum(((mags[key] - x[idx]) / errs[key]) ** 2)
		return s

	start = [np.median(mags[key]) for key in ufilters]
	error = [1. for _ in ufilters]
	name = ["f_magStar_" + key for key in ufilters]
	m_flat = Minuit.from_array_func(least_squares_flat,
									start=start,
									error=error,
									name=name,
									errordef=1,
									print_level=0)
	m_flat.migrad()
	global GLOBAL_COUNTER
	GLOBAL_COUNTER += 1
	# print(str(GLOBAL_COUNTER) + " : " + subdf.name)
	flat_params = m_flat.values

	# init for output
	flat_keys = ["f_magStar_" + key for key in COLOR_FILTERS.keys()]
	flat_values = []
	for key in COLOR_FILTERS.keys():
		if key in ufilters:
			flat_values.append(m_flat.values["f_magStar_" + key])
		else:
			flat_values.append(np.nan)
	flat_fmin = m_flat.get_fmin()
	flat_fval = m_flat.fval

	alltimes = np.concatenate(list(time.values()))
	bounds_simple = np.array([[-30, 30] for _ in ufilters] + [[0, 3], [alltimes.min()-100, alltimes.max()+100], [0, 3.7]])
	try:
		fval, pms, nbloops = diff_ev_lhs(to_minimize_simple_nd, list(time.values()), list(mags.values()),
									 list(errs.values()), bounds=bounds_simple, pop=70, recombination=0.3)
	except ZeroDivisionError:
		logging.error("Divison by zero in diffev. Current star: "+str(subdf.name))

	names = ["u0", "t0", "tE"] + ["magStar_" + key for key in COLOR_FILTERS.keys()]
	micro_keys = names

	pms = list(pms)

	def least_squares_microlens(x):
		lsq = 0
		for idx, key in enumerate(ufilters):
			lsq += np.sum(((mags[key] - microlens_simple(time[key], x[idx + 3], 0., x[0], x[1], x[2])) / errs[key]) ** 2)
		return lsq

	start = pms[-3:-1] + [np.power(10, pms[-1])] + pms[:-3]
	names = ["u0", "t0", "tE"] + ["magStar_" + key for key in ufilters]
	errors = [0.1, 100, 10] + [2 for key in ufilters]
	limits = [(0, 3), (alltimes.min()-100, alltimes.max()+100), (1, 5000)] + [(None, None) for _ in ufilters]
	m_micro = Minuit.from_array_func(least_squares_microlens,
									 start=start,
									 error=errors,
									 limit=limits,
									 name=names,
									 errordef=1,
									 print_level=0)

	m_micro.migrad()
	micro_params = m_micro.values

	micro_errors = []
	micro_error_labels = []

	if hesse:
		micro_error_labels = ["error_" + name for name in micro_keys]
		micro_errors = [np.nan]*len(micro_keys)
		m_micro.hesse()
		micro_errors = [m_micro.errors["u0"], m_micro.errors["t0"], m_micro.errors["tE"]]
		for key in COLOR_FILTERS.keys():
			if key in ufilters:
				micro_errors.append(m_micro.errors["magStar_" + key])
			else:
				micro_errors.append(np.nan)
	elif minos:
		for name in micro_keys:
			micro_error_labels+=["lower_error_"+name, "upper_error_"+name, "valid_lower_error_"+name, "valid_upper_error_"+name]
		micro_errors = [np.nan]*(3+len(COLOR_FILTERS))*4
		merrors=False
		try:
			merrors = m_micro.minos()
		except RuntimeError:
			print("Migrad did not converge properly on star " + str(subdf.name))
		if merrors:
			micro_errors = [merrors["u0"].lower, merrors["u0"].upper, merrors["u0"].lower_valid, merrors["u0"].upper_valid,
							merrors["t0"].lower, merrors["t0"].upper, merrors["t0"].lower_valid,
							merrors["t0"].upper_valid,
							merrors["tE"].lower, merrors["tE"].upper, merrors["tE"].lower_valid,
							merrors["tE"].upper_valid]
			for key in COLOR_FILTERS.keys():
				if key in ufilters:
					micro_errors += [merrors["magStar_" + key].lower, merrors["magStar_" + key].upper,
										 merrors["magStar_" + key].lower_valid, merrors["magStar_" + key].upper_valid]
				else:
					micro_errors+=[np.nan]*4

	lsqs = []
	micro_values = [micro_params['u0'], micro_params['t0'], micro_params['tE']]
	for key in COLOR_FILTERS.keys():
		if key in ufilters:
			lsqs.append(np.sum(((mags[key] - microlens_simple(time[key], micro_params["magStar_" + key], 0, micro_params['u0'], micro_params['t0'], micro_params['tE'])) / errs[key]) ** 2))
			micro_values.append(m_micro.values["magStar_" + key])
		else:
			lsqs.append(np.nan)
			micro_values.append(np.nan)
	micro_fmin = m_micro.get_fmin()
	micro_fval = m_micro.fval

	counts = []
	flat_chi2s = []
	for key in COLOR_FILTERS.keys():
		if key in ufilters:
			counts.append((~np.isnan(mags[key])).sum())
			flat_chi2s.append(np.sum(((mags[key] - m_flat.values["f_magStar_"+key]) / errs[key]) ** 2))
		else:
			counts.append(0)
			flat_chi2s.append(np.nan)

	return pd.Series(
		micro_values + [micro_fmin, micro_fval] + micro_errors
		+ flat_values + [flat_fmin, flat_fval]
		+ counts
		+ list(scount.values())
		+ lsqs
		+ flat_chi2s
		+ list(median_errors.values())
		+ list(intrinsic_dispersion.values())

		,

		index=micro_keys + ['micro_fmin', 'micro_fval'] + micro_error_labels
			  + flat_keys + ['flat_fmin', 'flat_fval']
			  + ["counts_" + key for key in COLOR_FILTERS.keys()]  # ["counts_RE", "counts_BE", "counts_RM", "counts_BM"]
			  + ["scounts_"+key for key in COLOR_FILTERS.keys()]
			  + ["micro_chi2_" + key for key in COLOR_FILTERS.keys()]  # ['micro_chi2_RE', 'micro_chi2_BE', 'micro_chi2_RM', 'micro_chi2_BM']
			  + ["flat_chi2_" + key for key in COLOR_FILTERS.keys()]  # ['flat_chi2_RE', 'flat_chi2_RM', 'flat_chi2_BE', 'flat_chi2_BM']
			  + ["magerr_" + key + "_median" for key, cf in COLOR_FILTERS.items()]  # ['errRE_median', 'errBE_median', 'errRM_median', 'errBM_median']
			  + ["intr_disp_" + key for key in intrinsic_dispersion.keys()]
	)


def fit_ml_de_blend(subdf, do_cut5=False, hesse=False, minos=False):
	"""Fit on one star

	Color filter names must be stocked in a COLOR_FILTERS dictionnary
	for example : COLOR_FILTERS = {"r":{"mag":"mag_r", "err":"magerr_r"},
								   "g":{"mag":"mag_g", "err":"magerr_g"}}

	Parameters
	----------
	subdf : pd.DataFrame
		Lightcurve data. Should have magnitudes stocked in "mag_*color*"  columns, magnitude errors in "magerr_*color*",
		for each *color* name and timestamps in "time" column

	do_cut5 : bool
		If True, clean aberrant points using distance from median of 5 points (default: {False})

	hesse : bool
		Use HESSE to compute paraoblic errors (not really valid as the chi2 profile is not parabolic, for tE)

	minos : bool
		Use MINOS to compute asymetric errors (slow the function, ~x1.5)

	Returns
	-------
	pd.Series
		Contains parameters for the microlensing and flat curve fits, their chi2, informations on the fitter (fmin) and dof :

		mulens Fit results parameters : mag_1, ... mag_n, u0, t0, tE
		mulens Minuit fit output informations
		mulens Fit final Chi^2
		flat Fit results parameters : mag_1, ... mag_n
		flat Minuit fit output informations
		flat Fit final Chi^2
		Number of points used in each color filter
		Individual final mulens fit chi^2 value for each filter
		Individual final flat fit chi^2 value for each filter
		Intrinsic dispersion for each color filter
	"""

	# print(subdf.name)

	mask = dict()
	errs = dict()
	mags = dict()
	cut5 = dict()
	time = dict()

	min_err = 0.0
	remove_extremities = True
	tolerance_ratio = 0.9
	p = True
	ufilters = []

	for key in COLOR_FILTERS.keys():
		mask[key] = subdf[COLOR_FILTERS[key]["mag"]].notnull() & subdf[COLOR_FILTERS[key]["err"]].notnull() & subdf[COLOR_FILTERS[key]["err"]].between(
			min_err, 9.999, inclusive=False)  # No nan and limits on errors

		if mask[key].sum()>2:		#Check if there are more than 3 valid points in the current color
			ufilters.append(key)
			mags[key] = subdf[mask[key]][COLOR_FILTERS[key]["mag"]]  # mags
			errs[key] = subdf[mask[key]][COLOR_FILTERS[key]["err"]]  # errs
			cut5[key] = np.abs((mags[key].rolling(5, center=True).median() - mags[key][2:-2])) / errs[key][2:-2] < 5

			if not remove_extremities:
				cut5[key][:2] = True
				cut5[key][-2:] = True

			p *= cut5[key].sum() / len(cut5[key]) < tolerance_ratio

	if do_cut5 and not p:
		for key in ufilters:
			time[key] = subdf[mask[key]][cut5[key]].time.to_numpy()
			errs[key] = errs[key][cut5[key]].to_numpy()
			mags[key] = mags[key][cut5[key]].to_numpy()
	else:
		for key in ufilters:
			time[key] = subdf[mask[key]].time.to_numpy()
			errs[key] = errs[key].to_numpy()
			mags[key] = mags[key].to_numpy()

	# Normalize errors
	intrinsic_dispersion = dict()
	median_errors = dict()
	scount = dict()
	for key in COLOR_FILTERS.keys():
		intrinsic_dispersion[key] = np.nan
		median_errors[key] = np.nan
		scount[key] = 0
	for key in ufilters:
		median_errors[key] = np.median(errs[key])
		if len(mags[key]) <= 3:
			intrinsic_dispersion[key] = 1.
		else:
			tmsk = errs[key]<0.6
			scount[key] = tmsk.sum()
			intrinsic_dispersion[key] = nb_truncated_intrinsic_dispersion(time[key][tmsk], mags[key][tmsk], errs[key][tmsk], fraction=0.0)
			if intrinsic_dispersion[key] > 0:
				errs[key] = errs[key] * intrinsic_dispersion[key]
			else:
				print(f"null intrinsic dispersion for {subdf.name}")

	# if magRE.size==0 or magBE.size==0 or magRM.size==0 or magBM.size==0:
	# 	return pd.Series(None)

	# flat fit
	def least_squares_flat(x):
		s = 0
		for idx, key in enumerate(ufilters):
			s += np.sum(((mags[key] - x[idx]) / errs[key]) ** 2)
		return s

	start = [np.median(mags[key]) for key in ufilters]
	error = [1. for _ in ufilters]
	name = ["f_magStar_" + key for key in ufilters]
	m_flat = Minuit.from_array_func(least_squares_flat,
									start=start,
									error=error,
									name=name,
									errordef=1,
									print_level=0)
	m_flat.migrad()
	global GLOBAL_COUNTER
	GLOBAL_COUNTER += 1
	# print(str(GLOBAL_COUNTER) + " : " + subdf.name)
	flat_params = m_flat.values

	# init for output
	flat_keys = ["f_magStar_" + key for key in COLOR_FILTERS.keys()]
	flat_values = []
	for key in COLOR_FILTERS.keys():
		if key in ufilters:
			flat_values.append(m_flat.values["f_magStar_" + key])
		else:
			flat_values.append(np.nan)
	flat_fmin = m_flat.get_fmin()
	flat_fval = m_flat.fval

	alltimes = np.concatenate(list(time.values()))
	bounds_blend = np.array([[-30, 30] for _ in ufilters] + [[0, 1] for _ in ufilters] + [[0, 3], [alltimes.min()-100, alltimes.max()+100], [0, 3]])
	try:
		fval, pms, nbloops = diff_ev_lhs(to_minimize_blended_nd, list(time.values()), list(mags.values()),
									 list(errs.values()), bounds=bounds_blend, pop=100, recombination=0.3)
	except ZeroDivisionError:
		logging.error("Divison by zero in diffev. Current star: "+str(subdf.name))

	names = ["u0", "t0", "tE"] + ["magStar_" + key for key in COLOR_FILTERS.keys()] + ["blend_" + key for key in COLOR_FILTERS.keys()]
	micro_keys = names

	pms = list(pms)

	nb_mags = len(ufilters)
	def least_squares_microlens(x):
		lsq = 0
		for idx, key in enumerate(ufilters):
			lsq += np.sum(((mags[key] - microlens_simple(time[key], x[idx + 3], x[nb_mags+3+idx], x[0], x[1], x[2])) / errs[key]) ** 2)
		return lsq

	start = pms[-3:-1] + [np.power(10, pms[-1])] + pms[:-3]
	names = ["u0", "t0", "tE"] + ["magStar_" + key for key in ufilters] + ["blend_" + key for key in ufilters]
	errors = [0.1, 10, 1] + [1 for _ in ufilters] + [0.05 for _ in ufilters]
	limits = [(0, 3), (alltimes.min()-100, alltimes.max()+100), (1, 1000)] + [(None, None) for _ in ufilters] + [(0, 1) for _ in ufilters]
	m_micro = Minuit.from_array_func(least_squares_microlens,
									 start=start,
									 error=errors,
									 limit=limits,
									 name=names,
									 errordef=1,
									 print_level=0)

	m_micro.migrad()
	micro_params = m_micro.values

	micro_errors = []
	micro_error_labels = []

	if hesse:
		micro_errors = dict()
		for name in micro_keys:
			if name in m_micro.errors:
				micro_errors["error_" + name] = m_micro.errors[name]
			else:
				micro_errors["error_" + name] = np.nan
		micro_error_labels = list(micro_errors.keys())
		micro_errors = list(micro_errors.values())
	elif minos:
		print("Not yet implemented")

	lsqs = []
	max_nb_mags = len(COLOR_FILTERS)
	micro_values = dict()
	micro_values["u0"] = micro_params['u0']
	micro_values["t0"] = micro_params['t0']
	micro_values["tE"] = micro_params['tE']
	for key in COLOR_FILTERS.keys():
		if key in ufilters:
			lsqs.append(np.sum(((mags[key] - microlens_simple(time[key], micro_params["magStar_" + key], micro_params["blend_" + key], micro_params['u0'], micro_params['t0'], micro_params['tE'])) / errs[key]) ** 2))
			micro_values["magStar_" + key] = m_micro.values["magStar_" + key]
			micro_values["blend_" + key] = m_micro.values["blend_" + key]
		else:
			lsqs.append(np.nan)
			micro_values["magStar_" + key] = np.nan
			micro_values["blend_" + key] = np.nan
	micro_keys = list(micro_values.keys())
	micro_values = list(micro_values.values())
	micro_fval = m_micro.fval
	micro_fmin = m_micro.get_fmin()


	counts = []
	flat_chi2s = []
	for key in COLOR_FILTERS.keys():
		if key in ufilters:
			counts.append((~np.isnan(mags[key])).sum())
			flat_chi2s.append(np.sum(((mags[key] - m_flat.values["f_magStar_"+key]) / errs[key]) ** 2))
		else:
			counts.append(0)
			flat_chi2s.append(np.nan)

	return pd.Series(
		list(micro_values) + [micro_fmin, micro_fval] + micro_errors
		+ flat_values + [flat_fmin, flat_fval]
		+ counts
		+ list(scount.values())
		+ lsqs
		+ flat_chi2s
		+ list(median_errors.values())
		+ list(intrinsic_dispersion.values())

		,

		index=micro_keys + ['micro_fmin', 'micro_fval'] + micro_error_labels
			  + flat_keys + ['flat_fmin', 'flat_fval']
			  + ["counts_" + key for key in COLOR_FILTERS.keys()]  # ["counts_RE", "counts_BE", "counts_RM", "counts_BM"]
			  + ["scounts_"+key for key in COLOR_FILTERS.keys()]
			  + ["micro_chi2_" + key for key in COLOR_FILTERS.keys()]  # ['micro_chi2_RE', 'micro_chi2_BE', 'micro_chi2_RM', 'micro_chi2_BM']
			  + ["flat_chi2_" + key for key in COLOR_FILTERS.keys()]  # ['flat_chi2_RE', 'flat_chi2_RM', 'flat_chi2_BE', 'flat_chi2_BM']
			  + ["magerr_" + key + "_median" for key, cf in COLOR_FILTERS.items()]  # ['errRE_median', 'errBE_median', 'errRM_median', 'errBM_median']
			  + ["intr_disp_" + key for key in intrinsic_dispersion.keys()]
	)
