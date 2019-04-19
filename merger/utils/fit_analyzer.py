import pandas as pd
from iminuit import Minuit
import numpy as np
import matplotlib.pyplot as plt


COLOR_FILTERS = {
	'red_E':{'mag':'red_E', 'err': 'rederr_E'},
	'red_M':{'mag':'red_M', 'err': 'rederr_M'},
	'blue_E':{'mag':'blue_E', 'err': 'blueerr_E'},
	'blue_M':{'mag':'blue_M', 'err': 'blueerr_M'}
}

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

def microlensing_amplification(t, u0, t0, tE):
	u = np.sqrt(u0*u0 + ((t-t0)**2)/tE/tE)
	return (u**2+2)/(u*np.sqrt(u**2+4))

def show(curr_id):
	fit_params = params.loc[curr_id]
	print(fit_params)
	g = merged.loc[curr_id]

	errRE = g.rederr_E
	errBE = g.blueerr_E
	cre = (errRE>0) & (errRE<9.999)
	cbe = (errBE>0) & (errBE<9.999)
	gBE= g[cbe] 
	gRE= g[cre] 

	fig, ax = plt.subplots(4,1,sharex=True)
	ax[0].errorbar(gRE.time, gRE.red_E, yerr=gRE.rederr_E, fmt='x', ecolor='black', elinewidth=0.5, color='gray', label='red EROS')
	ax[0].plot(gRE.time, fit_params.magStarRE-2.5*np.log10(microlensing_amplification(gRE.time, fit_params.u0, fit_params.t0, fit_params.tE)), color='black')

	ax[0].errorbar(gBE.time, gBE.blue_E, yerr=gBE.blueerr_E, fmt='x', ecolor='black', elinewidth=0.5, color='lightgreen', label='blue EROS')
	ax[0].plot(gBE.time, fit_params.magStarBE-2.5*np.log10(microlensing_amplification(gBE.time, fit_params.u0, fit_params.t0, fit_params.tE)), color='green')


	ax[0].invert_yaxis()
	ax[0].legend()
	ax[0].set_xlabel('Time (MJD)')
	ax[0].set_title('EROS')

	ax[1].errorbar(gRE.time, fit_params.magStarRE-2.5*np.log10(microlensing_amplification(gRE.time, fit_params.u0, fit_params.t0, fit_params.tE))-gRE.red_E, yerr=gRE.rederr_E, fmt='x', ecolor='black', elinewidth=0.5, color='gray', label='red EROS')
	ax[1].errorbar(gBE.time, fit_params.magStarBE-2.5*np.log10(microlensing_amplification(gBE.time, fit_params.u0, fit_params.t0, fit_params.tE))-gBE.blue_E, yerr=gBE.blueerr_E, fmt='x', ecolor='black', elinewidth=0.5, color='lightgreen', label='blue EROS')
	ax[1].axhline(0, color='black', linewidth=0.5)
	ax[1].set_ylim(-0.5, 0.5)

	errRM = g.rederr_M
	errBM = g.blueerr_M
	crm = (errRM>0) & (errRM<9.999)
	cbm = (errBM>0) & (errBM<9.999)
	gBM = g[cbm] 
	gRM = g[crm] 

	ax[2].errorbar(gRM.time, gRM.red_M, yerr=gRM.rederr_M, fmt='x', ecolor='black', elinewidth=0.5, color='red', label='red MACHO')
	ax[2].plot(gRM.time, fit_params.magStarRM-2.5*np.log10(microlensing_amplification(gRM.time, fit_params.u0, fit_params.t0, fit_params.tE)), color='red')

	ax[2].errorbar(gBM.time, gBM.blue_M, yerr=gBM.blueerr_M, fmt='x', ecolor='black', elinewidth=0.5, color='blue', label='blue MACHO')
	ax[2].plot(gBM.time, fit_params.magStarBM-2.5*np.log10(microlensing_amplification(gBM.time, fit_params.u0, fit_params.t0, fit_params.tE)), color='blue')

	ax[2].set_ylim(-10, 0)
	ax[2].invert_yaxis()
	ax[2].legend()
	ax[2].set_xlabel('Time (MJD)')
	ax[2].set_title('MACHO')

	ax[3].errorbar(gRM.time, fit_params.magStarRM-2.5*np.log10(microlensing_amplification(gRM.time, fit_params.u0, fit_params.t0, fit_params.tE))-gRM.red_M, yerr=gRM.rederr_M, fmt='x', ecolor='black', elinewidth=0.5, color='red', label='red MACHO')
	ax[3].errorbar(gBM.time, fit_params.magStarBM-2.5*np.log10(microlensing_amplification(gBM.time, fit_params.u0, fit_params.t0, fit_params.tE))-gBM.blue_M, yerr=gBM.blueerr_M, fmt='x', ecolor='black', elinewidth=0.5, color='blue', label='blue MACHO')
	ax[3].axhline(0, color='black', linewidth=0.5)
	ax[3].set_ylim(-0.5, 0.5)
	plt.show()

WORKING_DIR_PATH = "/Volumes/DisqueSauvegarde/working_dir/"
merged = pd.read_pickle(WORKING_DIR_PATH+"5_lm0103.pkl") #14_lm0594.pkl
params = pd.read_pickle(WORKING_DIR_PATH+"res_5_lm0103_nocut5.pkl")

if len(params)!=merged.id_E.nunique():
	print("Length mismatch between data and fit parameters !")

params['micro_valid'] = params.apply(lambda x: x.micro_fmin.is_valid, axis=1)
c1 = params.micro_valid
merged.set_index("id_E", inplace=True)
params["delta_chi2"] = (params.flat_fval**2-params.micro_fval**2)/params.micro_fval**2
params['full_delta_chi2'] = params.delta_chi2*(params.dof-7)

filtered = params[(params.delta_chi2>1) & (params.tE>301) & params.micro_valid].sort_values(by='delta_chi2')

for id_e in filtered.index:
	show(id_e)