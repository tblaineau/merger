import numpy as np
import matplotlib.pyplot as plt
import astropy.units as units
import time
from scipy.stats import rv_continuous

def pvT(v_T, v0=220):
	#Proba to find v_T i
	return (2*v_T/(v0**2))*np.exp(-v_T**2/(v0**2))

def vec_galactic(r, l, b):
	return np.array([
			r*np.cos(b)*np.cos(l),
			r*np.cos(b)*np.sin(l),
			r*np.sin(b)
		])

def rho_halo_slow(x, a=5000, rho_0=0.0079):
	#a: halo core radius (pc)
	#rho_0: local dark matter density (m_sol/pc**3)
	d_sol = 8500	#sun-galactic center distance (pc)
	l_lmc, b_lmc = 280.4652/180.*np.pi, -32.8884/180.*np.pi 	#galactic coordinates of LMC
	r_lmc = 55000	#lmc-sun distance
	x_lmc = vec_galactic(x*r_lmc, l_lmc, b_lmc) #sun-lmc vector
	x_gal_center = vec_galactic(d_sol, 359.94425*np.pi/180., -0.04616*np.pi/180.) #sun-galactic center vector
	if not np.isscalar(x):
		r = np.linalg.norm(x_lmc.T - x_gal_center, axis=1)
	else:
		r = np.linalg.norm(x_lmc - x_gal_center)
	return rho_0 * (a*a+d_sol*d_sol)/(a*a+r*r)

a=5000.
rho_0=0.0079
d_sol = 8500
l_lmc, b_lmc = 280.4652/180.*np.pi, -32.8884/180.*np.pi
r_lmc = 55000
cosb_lmc = np.cos(b_lmc)
cosl_lmc = np.cos(l_lmc)
d_lmc = np.sqrt(r_lmc**2 - 2*d_sol*r_lmc*cosl_lmc*cosb_lmc + d_sol**2)

def rho_halo(x, a=5000, rho_0=0.0079):
	r2 = (x*r_lmc)**2 - 2*x*r_lmc*d_sol*cosb_lmc*cosl_lmc + d_sol**2
	return rho_0 * (a*a+d_sol*d_sol)/(a*a+r2)

def rho(x, rho0=0.0079, a=5, r0=8.5):
	#in this function r_lmc = 48 kpc
	rho=rho0*(a**2+r0**2)/(a**2+(2303.24*(x**2)-(124.44*x)+72.25))
	return rho

def vt_ppf(x, v0=220):
	return np.sqrt(-np.log(1-x)*v0*v0)

def rdm(func, range_x, nb=1):
	k=0;
	v=[]
	min_x, max_x = range_x
	x = np.linspace(min_x, max_x, 100000)
	max_funcx = np.max(func(x))
	print(max_funcx)
	while len(v)<nb:
		x=np.random.uniform(min_x, max_x);
		y=np.random.uniform(0, max_funcx);
		if x!=0 and y<func(x):
			v.append(x)
	return v

def vT_rdm(nb):
	return rdm(pvT, nb)

def rho_halo_rdm(nb):
	return rdm(rho_halo, nb)

"""def ngc_pdf(d):
	A = rho_0*(d_sol**2+a**2)/a**2
	ngc = 1/(A*a*np.pi/2)
	return A*ngc*1/(d*d/(a*a)+1)

def ngc_cdf(d):
	# A = rho_0*(d_sol**2+a**2)/a**2
	# ngc = 1/(A*a*(np.arctan(d_lmc/a)-np.arctan(d_sol/a)))
	# return a*ngc*A*(np.arctan(d/a)-np.arctan(d_sol/a))	
	A = rho_0*(d_sol**2+a**2)/a**2
	ngc = 1/(A*a*np.pi/2)
	return a*ngc*A*np.arctan(d/a)

def ngc_ppf(q):
	# A = rho_0*(d_sol**2+a**2)/a**2
	# ngc = 1/(A*a*(np.arctan(d_lmc/a)-np.arctan(d_sol/a)))
	# return a*np.tan(q / (a*ngc*A) + np.arctan(d_sol/a))	
	A = rho_0*(d_sol**2+a**2)/a**2
	ngc = 1/(A*a*np.pi/2)
	return a*np.tan(q / (a*ngc*A))"""

def ngc_pdf(x):
	A = d_sol**2+a**2
	B = d_sol*cosb_lmc*cosl_lmc
	return rho_0*A/((x*r_lmc)**2-2*x*r_lmc*B+A)

def ngc_cdf(x):
	A = d_sol**2+a**2
	B = d_sol*cosb_lmc*cosl_lmc
	delta = np.sqrt(A-B*B)
	return rho_0*A/r_lmc/delta*(np.arctan((r_lmc*x-B)/delta)-np.arctan(-B/delta))

def ngc_ppf(rho):
	A = d_sol**2+a**2
	B = d_sol*cosb_lmc*cosl_lmc
	delta = np.sqrt(A-B*B)
	return (B + delta*np.tan(r_lmc*delta*rho/(rho_0*A)+np.arctan(-B/delta)))/r_lmc

def lenses_pdf(x):
	return ngc_pdf(x)*np.sqrt(x*(1-x))

def lenses_cdf(x):
	A = d_sol**2+a**2
	B = d_sol*cosb_lmc*cosl_lmc
	delta = np.sqrt(A-B*B)

	def pt1(x):
		return - (A-2*B**2)*np.arctan((r_lmc*x-B)/delta)/delta
	def pt2(x):
		return B*np.log(r_lmc*x*(r_lmc*x-2*B)+A)+r_lmc*x

	return A/r_lmc**3*((pt1(x)+pt2(x))-(pt1(0)+pt2(0)))

def d_to_x(d):
	a = 1
	b = -2*cosb_lmc*cosl_lmc
	c = d_sol**2-d**2
	return (-b+np.sqrt(b**2-4*a*c))/(2*a*r_lmc)

def x_to_d(x):
	return np.sqrt((x*r_lmc)**2 - 2*x*d_sol*r_lmc*cosl_lmc*cosb_lmc + d_sol**2)

class ngc_generator(rv_continuous):
	def _pdf(self, x):
		#A = rho_0*(d_sol**2+a**2)/a**2
		#ngc = 1/(A*a*(np.arctan(d_lmc/a)-np.arctan(d_sol/a)))
		#return A*ngc/((d/a)**2+1)
		return rho_halo(x)

class lenses_generator(rv_continuous):
	# x**2 * rho(x)
	
	def _pdf(self, x):
		return lenses_pdf(x)

	def _cdf(self, x):
		A = d_sol**2+a**2
		B = d_sol*cosb_lmc*cosl_lmc
		delta = np.sqrt(A-B*B)

		def pt1(x):
			return - (A-2*B**2)*np.arctan((r_lmc*x-B)/delta)/delta
		def pt2(x):
			return B*np.log(r_lmc*x*(r_lmc*x-2*B)+A)+r_lmc*x

		return A/r_lmc**3*((pt1(x)+pt2(x))-(pt1(0)+pt2(0)))

RANGE=(0, 1)
lensesgen = lenses_generator()
x = np.linspace(0, 1, 1000)
out = np.linspace(lensesgen.cdf(0), lensesgen.cdf(1), 100)
SIZE=100000
BINS=50
plt.plot(x, lenses_pdf(x))
plt.gca().twinx().plot(x, lenses_cdf(x), color='red')
print(max(lenses_pdf(x)))
plt.figure()
plt.plot(out, lensesgen.ppf(out))
plt.show()

RANGE=(0, 1)
x = np.linspace(0, 1, 10000)
SIZE=100000
BINS=50
lensesgen = lenses_generator()
s1=time.time()
rhordm = rdm(lenses_pdf, (0, 1), SIZE)
s2=time.time()
#rhoppfrdm = lensesgen.ppf(np.random.uniform(lensesgen.cdf(0), lensesgen.cdf(1),SIZE))
s3= time.time()
#plt.hist(rhoppfrdm, histtype='step', bins=BINS)
#plt.hist(rhordm, histtype='step', bins=BINS)
plt.xlabel(r"$x$")
print(s2-s1)
print(s3-s2)
plt.plot(x, lensesgen.cdf(x), color='red')
plt.gca().set_ylim(0)
plt.show()

# RANGE=(0, 1)
# x = np.linspace(0, 1, 10000)
# SIZE=1000000*100
# BINS=1000
# s2=time.time()
# rhoppfrdm = ngc_ppf(np.random.uniform(ngc_cdf(0), ngc_cdf(1),SIZE))
# s3= time.time()
# plt.hist(rhoppfrdm, histtype='step', bins=BINS)
# plt.xlabel(r"$x$")
# print(s3-s2)
# plt.gca().twinx().plot(x, rho_halo(x)/BINS*SIZE, color='red')
# plt.gca().set_ylim(0)
# plt.show()



RANGE=(0, 1000)
x = np.linspace(0, 1000, 10000)
SIZE=10000
BINS=500
#plt.plot(x, gen.ppf(x))
s1 = time.time()
vtrdm = rdm(pvT, (0,1000), SIZE)
s2=time.time()
vtppfrdm = vt_ppf(np.random.random(SIZE))
s3= time.time()
# plt.plot(x, gen.cdf(x))
print(len(vtrdm), max(vtrdm))
plt.hist(vtrdm, histtype='step', bins=BINS, range=RANGE)
plt.hist(vtppfrdm, histtype='step', bins=BINS, range=RANGE)
plt.xlabel(r"$v_T$")
print(s2-s1)
print(s3-s2)
plt.gca().twinx().plot(x, pvT(x), color='red')
plt.gca().set_ylim(0)
plt.show()