import pandas as pd
from iminuit import Minuit
import numpy as np
import matplotlib.pyplot as plt

WORKING_DIR_PATH = "/Volumes/DisqueSauvegarde/working_dir/"
simulated = pd.read_pickle(WORKING_DIR_PATH+"simulated_test_no_blend.pkl")
params = pd.read_pickle(WORKING_DIR_PATH+"res_simu_no_blend.pkl")

COLOR_FILTERS = {
	'red_E':{'mag':'red_E', 'err': 'rederr_E'},
	'red_M':{'mag':'red_M', 'err': 'rederr_M'},
	'blue_E':{'mag':'blue_E', 'err': 'blueerr_E'},
	'blue_M':{'mag':'blue_M', 'err': 'blueerr_M'}
}

def microlensing_amplification(t, u0, t0, tE):
	u = np.sqrt(u0*u0 + ((t-t0)**2)/tE/tE)
	return (u**2+2)/(u*np.sqrt(u**2+4))

def microlensing_total(t, mag, u0, t0, tE):
	return mag - 2.5*np.log10(microlensing_amplification(t, u0, t0, tE))


def generate_microlensing_parameters(seed, blending=False):
	tmin = 48928
	tmax = 52697
	seed = int(seed.replace('lm0', '').replace('k', '0').replace('l', '1').replace('m', '2').replace('n', '3'))
	np.random.seed(seed)
	u0 = np.random.uniform(0,1)
	tE = np.exp(np.random.uniform(6.21, 9.21))
	t0 = np.random.uniform(tmin-tE/2., tmax+tE/2.)
	blend_factors = {}
	for key in COLOR_FILTERS.keys():
		if blending:
			blend_factors[key]=np.random.uniform(0, 0.7)
		else:
			blend_factors[key]=0
	return u0, t0, tE, blend_factors

for row in params.itertuples():
	temp = simulated.loc[row.Index]
	errRM = temp.rederr_M
	errBM = temp.blueerr_M
	crm = (errRM>0) & (errRM<9.999)
	cbm = (errBM>0) & (errBM<9.999)
	fig1, ax1 = plt.subplots(2,1, sharex=True)
	print(row)
	print(generate_microlensing_parameters(row.Index))
	ax1[0].errorbar(temp.time, temp.blue_E, yerr=temp.blueerr_E, fmt='+', elinewidth=0.5, color='green', ecolor='black')
	ax1[0].errorbar(temp.time, temp.red_E, yerr=temp.rederr_E, fmt='+', elinewidth=0.5, color='black', ecolor='black')
	ax1[0].set_ylim(14,25)
	ax1[0].invert_yaxis()
	ax1[1].errorbar(temp[cbm].time, temp[cbm].blue_M, yerr=temp[cbm].blueerr_M, fmt='+', elinewidth=0.5, color='blue', ecolor='black')
	ax1[1].errorbar(temp[crm].time, temp[crm].red_M, yerr=temp[crm].rederr_M, fmt='+', elinewidth=0.5, color='red', ecolor='black')
	ax1[1].set_ylim(-10,0)
	ax1[1].invert_yaxis()

	fig2, ax2 = plt.subplots(2,1, sharex=True)
	ax2[0].errorbar(temp.time, temp.ampli_blue_E, yerr=temp.ampli_blueerr_E, fmt='+', elinewidth=0.5, color='green', ecolor='black')
	ax2[0].errorbar(temp.time, temp.ampli_red_E, yerr=temp.ampli_rederr_E, fmt='+', elinewidth=0.5, color='black', ecolor='black')
	ax2[0].set_ylim(14,25)
	ax2[0].invert_yaxis()
	ax2[0].plot(temp.time, microlensing_total(temp.time, temp.blue_E.mean(), *generate_microlensing_parameters(row.Index)[:3]))
	ax2[1].errorbar(temp[cbm].time, temp[cbm].ampli_blue_M, yerr=temp[cbm].ampli_blueerr_M, fmt='+', elinewidth=0.5, color='blue', ecolor='black')
	ax2[1].errorbar(temp[crm].time, temp[crm].ampli_red_M, yerr=temp[crm].ampli_rederr_M, fmt='+', elinewidth=0.5, color='red', ecolor='black')
	ax2[1].plot(temp.time, microlensing_total(temp.time, temp.blue_M.mean(), *generate_microlensing_parameters(row.Index)[:3]))
	ax2[1].set_ylim(-10,0)
	ax2[1].invert_yaxis()

	fig3, ax3 = plt.subplots(2,1,sharex=True)
	ax3[0].plot(temp.time, microlensing_total(temp.time, temp.blue_E.mean(), *generate_microlensing_parameters(row.Index)[:3]), linestyle='--', linewidth=0.5, color='lightgreen')
	ax3[0].scatter(temp.time, microlensing_total(temp.time, row.magStarBE, row.u0, row.t0, row.tE), linestyle='-', linewidth=0.5, color='green')
	ax3[0].set_ylim(14,25)
	ax3[0].invert_yaxis()
	ax3[1].plot(temp.time, microlensing_total(temp.time, temp.blue_M.mean(), *generate_microlensing_parameters(row.Index)[:3]), linestyle='--', linewidth=0.5, color='lightblue')
	ax3[1].scatter(temp.time, microlensing_total(temp.time, row.magStarBM, row.u0, row.t0, row.tE), linestyle='-', linewidth=0.5, color='blue')
	ax3[1].set_ylim(-10,0)
	ax3[1].invert_yaxis()
	# plt.show()
	plt.waitforbuttonpress()
	plt.cla()
