import numpy as np
import os
import pandas as pd
from IPython.display import clear_output
from merger.clean.libraries.merger_library import COLOR_FILTERS
from merger.clean.libraries.differential_evolution import microlens_simple
import matplotlib.pyplot as plt
import time
import scipy.stats

def update_progress(progress):
	bar_length = 40
	if isinstance(progress, int):
		progress = float(progress)
	if not isinstance(progress, float):
		progress = 0
	if progress < 0:
		progress = 0
	if progress >= 1:
		progress = 1

	block = int(round(bar_length * progress))

	clear_output(wait = True)
	text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
	print(text)

def load_results(dirpath):
	pds = []
	for (dirpath, dirnames, filenames) in os.walk(dirpath):
		for i, filename in enumerate(filenames):
			if filename[-4:]=='.pkl':
				pds.append(pd.read_pickle(os.path.join(dirpath,filename)))
			update_progress(i/len(filenames))
	return pd.concat(pds)

def compute_values(all_params, color_filters=COLOR_FILTERS):
	all_params["dof"] = sum([all_params["counts_" + key] for key in color_filters.keys()])
	all_params["scounts"] = sum([all_params["scounts_" + key] for key in color_filters.keys()])
	for afilter in color_filters.keys():
		all_params['reduced_micro_chi2_' + afilter] = all_params['micro_chi2_' + afilter] / (
					all_params['counts_' + afilter] - 1.75)
		all_params['reduced_flat_chi2_' + afilter] = all_params['flat_chi2_' + afilter] / (
					all_params['counts_' + afilter] - 1.75)
		all_params["delta_chi2_" + afilter] = (all_params['flat_chi2_' + afilter] - all_params[
			'micro_chi2_' + afilter]) / all_params['micro_chi2_' + afilter]
		all_params['full_delta_chi2_' + afilter] = all_params['delta_chi2_' + afilter] * np.sqrt(
			(all_params['counts_' + afilter] - 1.75) / 2.)
	all_params["delta_chi2"] = (all_params.flat_fval - all_params.micro_fval) / all_params.micro_fval
	all_params['full_delta_chi2'] = all_params.delta_chi2 * np.sqrt((all_params.dof - 7) / 2.)
	all_params['reduced_micro_chi2'] = all_params.micro_fval / (all_params.dof - 7)
	all_params['reduced_flat_chi2'] = all_params.flat_fval / (all_params.dof - 7)
	return all_params


import matplotlib.ticker as tkr


def microlensing_amplification(t, u0, t0, tE):
	u = np.sqrt(u0 * u0 + ((t - t0) ** 2) / tE / tE)
	return (u ** 2 + 2) / (u * np.sqrt(u ** 2 + 4))

already_loaded = []

MACHO_files_path = '/sps/hep/eros/users/blaineau/macho_fast_read/'
EROS_files_path = '/Volumes/DisqueSauvegarde/EROS/lightcurves/lm_ex/'

from irods.session import iRODSSession
from irods.exception import DataObjectDoesNotExist
import ssl


def read_irods_eros_lightcurve(id_E):
	"""
	Read one eros lightcurve
	:param id_E:
	:return:
	"""
	IRODS_ROOT = '/eros/data/eros2/lightcurves/lm/'
	try:
		env_file = os.environ['IRODS_ENVIRONMENT_FILE']
	except KeyError:
		env_file = os.path.expanduser('~/.irods/irods_environment.json')

	ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH, cafile=None, capath=None,
											 cadata=None)
	ssl_settings = {'ssl_context': ssl_context}
	with iRODSSession(irods_env_file=env_file, **ssl_settings) as session:
		irods_filepath = os.path.join(IRODS_ROOT, id_E[:5], id_E[:6], id_E[:7], id_E + ".time")
		if irods_filepath[-4:] == 'time':
			try:
				obj = session.data_objects.get(irods_filepath)
			except DataObjectDoesNotExist:
				print(f"iRods file not found : {irods_filepath}")
			with obj.open('r') as f:
				lc = {'time': [], 'red_E': [], 'rederr_E': [], 'blue_E': [], 'blueerr_E': [], 'id_E': []}
				for line in f.readlines()[4:]:
					line = line.decode().split()
					lc["time"].append(float(line[0]) + 49999.5)
					lc["red_E"].append(float(line[1]))
					lc["rederr_E"].append(float(line[2]))
					lc["blue_E"].append(float(line[3]))
					lc["blueerr_E"].append(float(line[4]))
					lc["id_E"].append(id_E)
			eros_lc = pd.DataFrame.from_dict(lc)
	return eros_lc


def load_lightcurves(param_line):
	"""Load lightcurve from fit results, stores it in already_loaded"""
	global already_loaded
	incache = True
	if isinstance(already_loaded, pd.DataFrame):
		lc = already_loaded[(already_loaded.id_E == param_line['id_E'])
							& (already_loaded.id_M == param_line['id_M'])]

	if len(already_loaded) == 0 or len(lc) == 0:
		print("Loading from irods")
		eros_id = param_line['id_E']
		macho_id = param_line['id_M'].split(':')

		# eros_lc = read_eros_lighcurve(os.path.join(EROS_files_path,  eros_id[:5], eros_id[:6], eros_id[:7], eros_id+".time"))
		eros_lc = read_irods_eros_lightcurve(eros_id)

		st1 = time.time()
		macho_lcs = pd.read_parquet(os.path.join(MACHO_files_path,
												 "F_" + str(macho_id[0]),
												 "F_" + str(macho_id[0]) + "_" + str(macho_id[1]) + ".parquet"))
		print(time.time() - st1)
		macho_lc = macho_lcs[macho_lcs.id_M == ':'.join(macho_id)]
		del macho_lcs

		lc = pd.concat([macho_lc, eros_lc], sort=True).reset_index(drop=True)  # .drop(['id_E', 'id_M'], axis='columns')
		lc['id_E'] = param_line['id_E']
		lc['id_M'] = param_line['id_M']
		incache = False

	if len(already_loaded) == 0:
		already_loaded = lc
	elif not incache:
		already_loaded = pd.concat([already_loaded, lc])
	return lc


def visualize_curve(fit_params, g, expended=False, blind=True, figsize=(10, 10), marker="x", parallax=False,
					xlim=False, ylim=None, ylim_E=False, ylim_M=False):
	"""Print lightcurve from fit results and data"""
	errRE = g.rederr_E
	errBE = g.blueerr_E
	cre = (errRE > 0) & (errRE < 9.999)
	cbe = (errBE > 0) & (errBE < 9.999)
	gBE = g[cbe]
	gRE = g[cre]

	time = np.arange(g.time.min(), g.time.max(), 1)

	fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize)
	# maxRE = (gRE.set_index(pd.to_datetime(gRE.time, unit='D', origin='17-11-1858', cache=True)).sort_index()['red_E'].rolling('100D', closed='both').mean()).min()
	# ax[0].axhline(maxRE, color='red')

	# gBE_time_stmp = pd.to_datetime(gBE.time, unit='D', origin='17-11-1858', cache=True)
	# gRE_time_stmp = pd.to_datetime(gRE.time, unit='D', origin='17-11-1858', cache=True)
	# time_stmp = pd.to_datetime(time, unit='D', origin='17-11-1858', cache=True)
	# print(gBE_time_stmp)

	ax[0].errorbar(gRE.time, gRE.red_E, yerr=gRE.rederr_E, fmt=marker, ecolor='black', elinewidth=0.5, color='gray',
				   label='red EROS')
	ax[0].errorbar(gBE.time, gBE.blue_E, yerr=gBE.blueerr_E, fmt=marker, ecolor='black', elinewidth=0.5, color='green',
				   label='blue EROS')

	if not blind and not parallax:
		ax[0].plot(time, fit_params.magStar_blue_E - 2.5 * np.log10(
			microlensing_amplification(time, fit_params.u0, fit_params.t0, fit_params.tE)), color='green', lw=0.5)
		ax[0].plot(time, fit_params.magStar_red_E - 2.5 * np.log10(
			microlensing_amplification(time, fit_params.u0, fit_params.t0, fit_params.tE)), color='black', lw=0.5)
	elif not blind and parallax:
		ax[0].plot(time, parallax_microlensing_event(time, fit_params.magStar_blue_E, fit_params.u0, fit_params.t0,
													 fit_params.tE, fit_params.delta_u, fit_params.theta),
				   color='green', lw=0.5)
		ax[0].plot(time, parallax_microlensing_event(time, fit_params.magStar_red_E, fit_params.u0, fit_params.t0,
													 fit_params.tE, fit_params.delta_u, fit_params.theta),
				   color='black', lw=0.5)

	ax[0].invert_yaxis()
	ax[0].legend()
	ax[0].set_title('EROS')

	errRM = g.rederr_M
	errBM = g.blueerr_M
	crm = (errRM > 0) & (errRM < 9.999)
	cbm = (errBM > 0) & (errBM < 9.999)
	gBM = g[cbm]
	gRM = g[crm]

	ax[1].errorbar(gRM.time, gRM.red_M, yerr=gRM.rederr_M, fmt=marker, ecolor='black', elinewidth=0.5, color='red',
				   label='red MACHO')
	ax[1].errorbar(gBM.time, gBM.blue_M, yerr=gBM.blueerr_M, fmt=marker, ecolor='black', elinewidth=0.5, color='blue',
				   label='blue MACHO')

	if not blind:
		ax[1].plot(time, fit_params.magStar_red_M - 2.5 * np.log10(
			microlensing_amplification(time, fit_params.u0, fit_params.t0, fit_params.tE)), color='red', lw=0.5)
		ax[1].plot(time, fit_params.magStar_blue_M - 2.5 * np.log10(
			microlensing_amplification(time, fit_params.u0, fit_params.t0, fit_params.tE)), color='blue', lw=0.5)

	if xlim:
		ax[0].set_xlim(xlim)
		ax[1].set_xlim(xlim)

	if ylim_E:
		ax[0].set_ylim(ylim_E)
	else:
		ax[0].set_ylim(np.nanmax(np.concatenate([gRE.red_E.values, gBE.blue_E.values])),
					   np.nanmin(np.concatenate([gRE.red_E.values, gBE.blue_E.values])))

	# ax[1].set_ylim(-10, 0)
	ax[1].invert_yaxis()
	ax[1].legend()
	ax[1].set_title('MACHO')

	if ylim_M:
		ax[1].set_ylim(ylim_M)
	else:
		ax[1].set_ylim(np.nanmax(np.concatenate([gRM.red_M.values, gBM.blue_M.values])),
					   np.nanmin(np.concatenate([gRM.red_M.values, gBM.blue_M.values])))

	if ylim == "equal":
		temp = np.array([np.nanquantile(np.concatenate(i), [0.1, 0.9]) for i in
						 [[gRE.red_E.values, gBE.blue_E.values], [gRM.red_M.values, gBM.blue_M.values]]])
		lowq, highq = temp[:, 0], temp[:, 1]
		del temp
		off = np.max(highq - lowq)
		l1 = np.nanmedian(np.concatenate([gRE.red_E.values, gBE.blue_E.values]))  # np.max(highq[0])
		l2 = np.nanmedian(np.concatenate([gRM.red_M.values, gBM.blue_M.values]))  # np.max(highq[0])
		ax[0].set_ylim(l1 + off, l1 - off)
		ax[1].set_ylim(l2 + off, l2 - off)

	ax0 = ax[0].twiny()
	ax0.set_xlim(ax[0].get_xlim())
	ax0.xaxis.set_minor_locator(tkr.MultipleLocator(100))
	ax1 = ax[1].twiny()
	ax1.set_xlim(ax[1].get_xlim())
	ax1.xaxis.set_minor_locator(tkr.MultipleLocator(100))
	ax2 = ax[2].twiny()
	ax2.set_xlim(ax[2].get_xlim())
	ax2.xaxis.set_minor_locator(tkr.MultipleLocator(100))
	ax[0].xaxis.set_major_locator(tkr.FixedLocator(np.arange(48988, 53371, 365.25)))
	ax[0].xaxis.set_major_formatter(tkr.FixedFormatter(np.arange(1992, 2005, 1)))
	ax0.set_xlabel('Time (MJD)')
	ax[2].set_xlabel('Time (year)')

	# start, end = ax[1].get_xlim()
	# ax[1].set_xticks(np.arange(start, end, 365.25))
	ax[0].grid(axis='x', lw='1')
	ax0.grid(axis='x', which='minor', lw='0.5', ls='--')
	ax[1].grid(axis='x', lw='1')
	ax1.grid(axis='x', which='minor', lw='0.5', ls='--')
	ax[0].grid(axis='x', which='minor', lw='0.5', ls='--')
	ax[1].grid(axis='x', which='minor', lw='0.5', ls='--')

	ax[2].errorbar(gRE.time, -gRE.red_E + (fit_params.magStar_red_E - 2.5 * np.log10(
		microlensing_amplification(gRE.time, fit_params.u0, fit_params.t0, fit_params.tE))), yerr=gRE.rederr_E,
				   fmt=marker, ecolor='black', elinewidth=0.5, color='gray', label='red EROS')
	ax[2].errorbar(gBE.time, -gBE.blue_E + (fit_params.magStar_blue_E - 2.5 * np.log10(
		microlensing_amplification(gBE.time, fit_params.u0, fit_params.t0, fit_params.tE))), yerr=gBE.blueerr_E,
				   fmt=marker, ecolor='black', elinewidth=0.5, color='green', label='blue EROS')
	ax[2].errorbar(gRM.time, -gRM.red_M + (fit_params.magStar_red_M - 2.5 * np.log10(
		microlensing_amplification(gRM.time, fit_params.u0, fit_params.t0, fit_params.tE))), yerr=gRM.rederr_M,
				   fmt=marker, ecolor='black', elinewidth=0.5, color='red', label='red MACHO')
	ax[2].errorbar(gBM.time, -gBM.blue_M + (fit_params.magStar_blue_M - 2.5 * np.log10(
		microlensing_amplification(gBM.time, fit_params.u0, fit_params.t0, fit_params.tE))), yerr=gBM.blueerr_M,
				   fmt=marker, ecolor='black', elinewidth=0.5, color='blue', label='blue EROS')

	range_residual = 0.1  # np.nanmax([fit_params.intr_disp_blue_E, fit_params.intr_disp_blue_M, fit_params.intr_disp_red_E, fit_params.intr_disp_red_M])
	ax[2].set_ylim(-range_residual, range_residual)
	ax[2].axhline(0, c="k", ls="--")

	# plt.show()

	"""textr = f"ID EROS : {fit_params.id_E}\n ID MACHO : {fit_params.id_M}\n \
	$u_0$ : {fit_params.u0:0.2f}\n $t_E$ : {fit_params.tE:0.2f}, \n$t_0$ : {fit_params.t0:0.0f}"
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	ax[0].text(0.05, 0.95, textr, transform=ax[0].transAxes, fontsize=14, verticalalignment='top', bbox=props)"""

	if expended:
		plt.figure()
		plt.scatter(gRE.red_E, gRE.rederr_E, marker="+", s=10, color='black')
		plt.scatter(gBE.blue_E, gBE.blueerr_E, marker="+", s=10, color='green')
		plt.title("EROS")
		plt.figure()
		plt.scatter(gRM.red_M, gRM.rederr_M, marker="+", s=10, color='red')
		plt.scatter(gBM.blue_M, gBM.blueerr_M, marker="+", s=10, color='blue')
		plt.title("MACHO")
	plt.show()


def rolling_sum(a, n=4):
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:]


def weighted_moving_average(a, w, n=3):
	v = []
	for i in range(len(a) - n + 1):
		v.append(np.sum(a[i:i + n] * w[i:i + n]) / np.sum(w[i:i + n]))
	return v


def moving_median(a, n=3):
	v = []
	for i in range(len(a) - n + 1):
		v.append(np.median(a[i:i + n]))
	return v


def moving_average(a, n=3):
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n


colors = {"red_E": "black",
		  "red_M": "red",
		  "blue_E": "green",
		  "blue_M": "blue"
		  }


def visualize_curve_norm(fit_params, subdf, blind=True, figsize=(30, 15), nstat=20, marker=".", markersize=3, xlim=[]):
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
	do_cut5 = True

	for key in COLOR_FILTERS.keys():
		mask[key] = subdf[COLOR_FILTERS[key]["mag"]].notnull() & subdf[COLOR_FILTERS[key]["err"]].notnull() & subdf[
			COLOR_FILTERS[key]["err"]].between(
			min_err, 9.999, inclusive=False)  # No nan and limits on errors

		if mask[key].sum() > 2:  # Check if there are more than 3 valid points in the current color
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

	chi2s = dict()
	for key in COLOR_FILTERS.keys():
		if key in ufilters:
			chi2s[key] = ((mags[key] - microlens_simple(time[key], fit_params["magStar_" + key], 0, fit_params.u0,
														fit_params.t0, fit_params.tE)) / errs[key] / fit_params[
							  "intr_disp_" + key]) ** 2

	times = np.concatenate(list(time.values()))
	chi2s = np.concatenate(list(chi2s.values()))

	sidx = np.argsort(times)
	times = times[sidx]
	chi2s = chi2s[sidx]

	fig, ax = plt.subplots(3, 1, sharex=True, figsize=(30, 15), squeeze=False)

	for key in ufilters:
		ax[0, 0].errorbar(time[key] / 365.25, mags[key] - fit_params["magStar_" + key], yerr=errs[key], fmt=marker,
						  ecolor=colors[key], elinewidth=0.5, color=colors[key], label=key, markersize=markersize)

	fitr = -2.5 * np.log10(microlensing_amplification(times, fit_params.u0, fit_params.t0, fit_params.tE))
	fittruth = -fit_params["magStar_" + key] + microlens_simple(times, fit_params["magStar_" + key],
																fit_params["blend_" + key], fit_params.u0_true,
																fit_params.t0_true, fit_params.tE_true)
	if not blind:
		ax[0, 0].plot(times / 365.25, fitr, color='grey', lw=1.5)
		ax[0, 0].plot(times / 365.25, fittruth, color='grey', ls="--", lw=1.5)

	allv = np.concatenate([mags[key] - fit_params["magStar_" + key] for key in ufilters])
	# ax[0, 0].set_ylim(allv.min(), allv.max())
	maxmagerr = np.max([fit_params["magerr_" + key + "_median"] for key in COLOR_FILTERS.keys()])
	ax[0, 0].set_ylim(np.min([fitr.min() - maxmagerr * 2, -maxmagerr * 2]), +maxmagerr * 2)
	if len(xlim) == 2:
		ax[0, 0].set_xlim(xlim)
	ax[0, 0].invert_yaxis()

	rs = rolling_sum(chi2s, n=nstat)
	sigma = 3
	ci = scipy.stats.chi2(1).cdf(sigma ** 2)
	ax[2, 0].axhline(scipy.stats.chi2.ppf(ci, nstat) / nstat, c='r', ls='-', alpha=0.5)
	ax[2, 0].scatter(times[int(nstat / 2) - 1:-int(nstat / 2)] / 365.25, rs / nstat, c="r", marker=".")
	ax[2, 0].axhline(1, c='k', ls='-')
	ax[2, 0].axhline(np.sum(chi2s) / len(times), c='red', ls='--')

	ax[1, 0].scatter(times[int(nstat / 2) - 1:-int(nstat / 2)] / 365.25, moving_median(allv[sidx], nstat), c='k', s=1)
	ax[1, 0].plot(times / 365.25, fitr, color='grey', lw=1.5)
	ax[1, 0].invert_yaxis()
	plt.show()


#    df = pd.DataFrame(np.array([times, mags, errs]).T, columns=["times", "mags", "errs"])
#    df.rolling(10).apply(compute_rolling_chi2)


def loadvis_curve(params, norm=False, blind=True, marker='x', **kwargs):
	lc = load_lightcurves(params)
	if norm:
		visualize_curve_norm(params, lc, blind=blind, **kwargs)
	else:
		visualize_curve(params, lc, blind=blind, marker=marker, **kwargs)