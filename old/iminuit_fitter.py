import pandas as pd
import numpy as np
from iminuit import Minuit
import time
import matplotlib.pyplot as plt

GLOBAL_COUNTER=0
PERIOD_EARTH = 365.2422

def microlensing_event(t, u0, t0, tE, mag1):
	u = np.sqrt(u0*u0 + ((t-t0)**2)/tE/tE)
	return -2.5*np.log10((u**2+2)/(u*np.sqrt(u**2+4)))+mag1 

def parallax(t, mag, u0, t0, tE, delta_u, theta):
	PERIOD_EARTH = 365.2422
	alphaS = 80.8941667*np.pi/180.
	deltaS = -69.7561111*np.pi/180.
	epsilon = (90. - 66.56070833)*np.pi/180.
	t_origin = 58747 #(21 septembre 2019)
	sin_beta = np.cos(epsilon)*np.sin(deltaS) - np.sin(epsilon)*np.cos(deltaS)*np.sin(alphaS)
	beta = np.arcsin(sin_beta)
	lambda_star = np.arcsin((np.sin(epsilon)*np.sin(deltaS)+np.cos(epsilon)*np.sin(alphaS)*np.cos(deltaS))/np.cos(beta))
	tau = (t-t0)/tE
	phi = 2*np.pi * (t-t_origin)/PERIOD_EARTH - lambda_star
	u_D = np.array([ 
		-u0*np.sin(theta) + tau*np.cos(theta),
		 u0*np.cos(theta) + tau*np.sin(theta)
		])
	u_t = np.array([
		-delta_u*np.sin(phi),
		 delta_u*np.cos(phi)*sin_beta
		])
	u = np.linalg.norm(u_D-u_t, axis=0)
	return (u**2+2)/(u*np.sqrt(u**2+4))

def parallax_microlensing_event(t, mag, u0, t0, tE, delta_u, theta):
	return mag + - 2.5*np.log10(parallax(t, mag, u0, t0, tE, delta_u, theta))

def parallax_blend_microlensing_event(t, mag, blend, u0, t0, tE, delta_u, theta):
	return - 2.5*np.log10(blend*np.power(10, mag/-2.5) + (1-blend)*np.power(10, mag/-2.5) * parallax(t, mag, u0, t0, tE, delta_u, theta))

def fit_ml(subdf):
	#sélection des données, pas plus de 10% du temps de calcul en moyenne (0.01s vs 0.1s)
	#le fit peut durer jusqu'à 0.7s ou aussi rapide que 0.04s (en général False)
	# subdf.drop(subdf.red_E.nsmallest(5).index, inplace=True)
	# subdf.drop(subdf.blue_E.nsmallest(5).index, inplace=True)
	# subdf.drop(subdf.red_M.nsmallest(5).index, inplace=True)
	# subdf.drop(subdf.blue_M.nsmallest(5).index, inplace=True)

	maskRE = subdf.red_E.notnull() & subdf.rederr_E.notnull()
	maskBE = subdf.blue_E.notnull() & subdf.blueerr_E.notnull()
	maskRM = subdf.red_M.notnull() & subdf.rederr_M.notnull()
	maskBM = subdf.blue_M.notnull() & subdf.blueerr_M.notnull()

	errRE = subdf[maskRE].rederr_E
	errBE = subdf[maskBE].blueerr_E
	errRM = subdf[maskRM].rederr_M
	errBM = subdf[maskBM].blueerr_M

	min_err = 0.
	max_err = 9.999
	cre = errRE.between(min_err, max_err, inclusive=False)
	cbe = errBE.between(min_err, max_err, inclusive=False)
	crm = errRM.between(min_err, max_err, inclusive=False)
	cbm = errBM.between(min_err, max_err, inclusive=False)

	magRE = subdf[maskRE][cre].ampli_red_E
	magBE = subdf[maskBE][cbe].ampli_blue_E
	magRM = subdf[maskRM][crm].ampli_red_M
	magBM = subdf[maskBM][cbm].ampli_blue_M

	# magRE = subdf[maskRE][cre].red_E
	# magBE = subdf[maskBE][cbe].blue_E
	# magRM = subdf[maskRM][crm].red_M
	# magBM = subdf[maskBM][cbm].blue_M

	errRE = errRE[cre]
	errBE = errBE[cbe]
	errRM = errRM[crm]
	errBM = errBM[cbm]

	# cut5RE = np.abs((magRE.rolling(5, center=True).median()-magRE[2:-2]))/errRE[2:-2]<5
	# cut5BE = np.abs((magBE.rolling(5, center=True).median()-magBE[2:-2]))/errBE[2:-2]<5
	# cut5RM = np.abs((magRM.rolling(5, center=True).median()-magRM[2:-2]))/errRM[2:-2]<5
	# cut5BM = np.abs((magBM.rolling(5, center=True).median()-magBM[2:-2]))/errBM[2:-2]<5

	if True:#(cut5RE.sum()<20 and cut5BE.sum()<20) or (cut5BM.sum()<20 and cut5RM.sum()<20):
		timeRE = subdf[maskRE][cre].time.values
		timeBE = subdf[maskBE][cbe].time.values
		timeRM = subdf[maskRM][crm].time.values
		timeBM = subdf[maskBM][cbm].time.values

		errRE = errRE.values
		errBE = errBE.values
		errRM = errRM.values
		errBM = errBM.values

		magRE = magRE.values
		magBE = magBE.values
		magRM = magRM.values
		magBM = magBM.values
		#print("too much")

	else:
		timeRE = subdf[maskRE][cre][cut5RE].time.values
		timeBE = subdf[maskBE][cbe][cut5BE].time.values
		timeRM = subdf[maskRM][crm][cut5RM].time.values
		timeBM = subdf[maskBM][cbm][cut5BM].time.values

		errRE = errRE[cut5RE].values
		errBE = errBE[cut5BE].values
		errRM = errRM[cut5RM].values
		errBM = errBM[cut5BM].values

		magRE = magRE[cut5RE].values
		magBE = magBE[cut5BE].values
		magRM = magRM[cut5RM].values
		magBM = magBM[cut5BM].values

	# def least_squares_microlens(u0, t0, tE, magStarRE, magStarBE, magStarRM, magStarBM):
	# 	lsq1 = np.sum(((magRE - microlensing_event(timeRE, u0, t0, tE, magStarRE))/ errRE)**2)
	# 	lsq2 = np.sum(((magBE - microlensing_event(timeBE, u0, t0, tE, magStarBE))/ errBE)**2)
	# 	lsq3 = np.sum(((magRM - microlensing_event(timeRM, u0, t0, tE, magStarRM))/ errRM)**2)
	# 	lsq4 = np.sum(((magBM - microlensing_event(timeBM, u0, t0, tE, magStarBM))/ errBM)**2)
	# 	return lsq1+lsq2+lsq3+lsq4

	def least_squares_flat(f_magStarRE, f_magStarBE, f_magStarRM, f_magStarBM):
		return np.sum(((magRE - f_magStarRE)/errRE)**2) + np.sum(((magRM - f_magStarRM)/errRM)**2) + np.sum(((magBE - f_magStarBE)/errBE)**2) + np.sum(((magBM - f_magStarBM)/errBM)**2)

	def ls_parallax(params):
		u0, t0, tE, magStarRE, magStarBE, magStarRM, magStarBM, delta_u, theta = params
		lsq1 = np.sum(((magRE - parallax_microlensing_event(timeRE, magStarRE, u0, t0, tE, delta_u, theta))/ errRE)**2)
		lsq2 = np.sum(((magBE - parallax_microlensing_event(timeBE, magStarBE, u0, t0, tE, delta_u, theta))/ errBE)**2)
		lsq3 = np.sum(((magRM - parallax_microlensing_event(timeRM, magStarRM, u0, t0, tE, delta_u, theta))/ errRM)**2)
		lsq4 = np.sum(((magBM - parallax_microlensing_event(timeBM, magStarBM, u0, t0, tE, delta_u, theta))/ errBM)**2)
		return lsq1+lsq2+lsq3+lsq4

	# m_micro = Minuit(least_squares_microlens, 
	# 	u0=1., 
	# 	t0=50000, 
	# 	tE=500, 
	# 	magStarRE=magRE.mean(), 
	# 	magStarBE=magBE.mean(), 
	# 	magStarRM=magRM.mean(), 
	# 	magStarBM=magBM.mean(), 
	# 	error_u0=0.1, 
	# 	error_t0=5000, 
	# 	error_tE=100, 
	# 	error_magStarRE=2, 
	# 	error_magStarBE=2., 
	# 	error_magStarRM=2., 
	# 	error_magStarBM=2., 
	# 	limit_u0=(0,2), 
	# 	limit_tE=(300, 10000),
	# 	limit_t0=(40000, 60000),#(48927, 52698)
	# 	errordef=1,
	# 	print_level=0)

	params_names = ["u0", "t0", "tE", "magStarRE", "magStarBE", "magStarRM", "magStarBM", "delta_u", "theta"]
	params_init = {
		"u0":0.5, 
		"t0":50000, 
		"tE":1000, 
		"magStarRE":magRE.mean(), 
		"magStarBE":magBE.mean(), 
		"magStarRM":magRM.mean(), 
		"magStarBM":magBM.mean(), 
		"delta_u":0,
		"theta":0,
		"error_u0":0.1, 
		"error_t0":5000, 
		"error_tE":100, 
		"error_magStarRE":2, 
		"error_magStarBE":2., 
		"error_magStarRM":2., 
		"error_magStarBM":2., 
		"error_delta_u":0.05,
		"error_theta":0.5,
		"limit_u0":(0,2), 
		"limit_tE":(-10000, 10000),
		"limit_t0":(40000, 60000),
		"limit_delta_u":(0, None),
		"limit_theta":(0, 2*np.pi)
	}
	m_micro = Minuit(ls_parallax, 
		forced_parameters=params_names, 
		print_level=0, 
		errordef=1, 
		use_array_call=True,
		 **params_init)

	m_flat = Minuit(least_squares_flat, 
		f_magStarRE=20, 
		f_magStarBE=20, 
		f_magStarRM=-4, 
		f_magStarBM=-4, 
		error_f_magStarRE=2, 
		error_f_magStarBE=2., 
		error_f_magStarRM=2., 
		error_f_magStarBM=2., 
		errordef=1,
		print_level=0
		)

	m_micro.migrad()
	m_flat.migrad()

	global GLOBAL_COUNTER
	GLOBAL_COUNTER+=1
	print(str(GLOBAL_COUNTER)+" : "+subdf.id_M.iloc[0]+" "+str(m_micro.get_fmin().is_valid)+"     ", end='\r')
	return pd.Series(

		m_micro.values.values()+[m_micro.get_fmin(), m_micro.fval] 
		+ 
		m_flat.values.values()+[m_flat.get_fmin(), m_flat.fval]
		+[len(errRE)+len(errBE)+len(errRM)+len(errBM)], 

		index=m_micro.values.keys()+['micro_fmin', 'micro_fval']
		+
		m_flat.values.keys()+['flat_fmin', 'flat_fval']
		+["dof"]

		)

WORKING_DIR_PATH = "/Volumes/DisqueSauvegarde/working_dir/"
merged = pd.read_pickle(WORKING_DIR_PATH+'simulated_50_lm0220_parallax.pkl')
merged.replace(to_replace=[99.999,-99.], value=np.nan, inplace=True)
merged.dropna(axis=0, how='all', subset=['blue_E', 'red_E', 'blue_M', 'red_M'], inplace=True)

print("FILES LOADED")

start = time.time()
merged.reset_index(drop=True, inplace=True)
res= merged.groupby("id_E").apply(fit_ml)
end= time.time()
res.to_pickle(WORKING_DIR_PATH+'res_simulated_50_lm0220_parallax_nocut5.pkl')
print(str(end-start)+" seconds elapsed.")
print(str(len(res))+" stars fitted.")