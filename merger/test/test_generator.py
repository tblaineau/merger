import numpy as np
import time
import matplotlib.pyplot as plt
import numba as nb
from merger.old.parameter_generator import metropolis_hastings, randomizer_gauss, pdf_xvt, pdf_tEdu, tE_from_xvt, delta_u_from_x, f_vt, quad, p_xvt, rho_halo

def test_generator(mass, nb_samples=100000, save=False, load=False):
	x = np.linspace(0,1, 100)
	vt = np.linspace(0,1000,100)
	nb_bins = 100
	st1 = time.time()
	s = metropolis_hastings(pdf_xvt, randomizer_gauss, nb_samples, np.array([0.01, 0]), kwargs={'mass':mass})
	print(time.time()-st1)
	if save:
		np.save('xvt_samples.npy', s)
	if save or load:
		s = np.load('xvt_samples.npy')
	xrdm = s[:,0]
	vtrdm = s[:,1]

	RANGE = ((0,1), (-1000,1000))
	fig, axs = plt.subplots(2,2, sharex='col', sharey='row')
	axs[0,0].hist(xrdm, bins=nb_bins, histtype='step', color='red', range=RANGE[0])
	tmpa = axs[0,0].twinx()
	tmpa.plot(x, p_xvt(x,110, mass=mass)*rho_halo(x)/mass*x*x)
	tmpa.set_ylim(0)
	axs[1,0].hist2d(xrdm, vtrdm, bins=100, range=RANGE)
	#plt.contour(x, vT, pdf_xvt(x[None, :], vT[:, None], mass))
	axs[1,0].set_xlabel(r'$x$')
	axs[1,0].set_ylabel(r'$v_T [km/s]$')
	tempvt = [f_vt(vti) * quad(p_xvt, a=0, b=1, args=(vti, mass))[0] for vti in vt]
	axs[1,1].hist(vtrdm, bins=nb_bins, histtype='step', color='red', orientation='horizontal', range=RANGE[1])
	tmpa = axs[1,1].twiny()
	tmpa.plot(tempvt, vt)
	tmpa.set_xlim(0)
	tmpa.set_ylim(0)
	plt.show()

	durdm = delta_u_from_x(xrdm, mass=mass)
	terdm = tE_from_xvt(xrdm, vtrdm, mass=mass)

	#plt.hist2d(durdm, terdm, bins=100, range=((0, 0.02), (0, 3000)))
	delta_u = np.linspace(0, 0.02, 100)
	tE = np.linspace(0, 3000, 100)
	ptEdu = pdf_tEdu(tE[:,None], delta_u[None,:], mass=mass)
	plt.contour(delta_u, tE, ptEdu)
	plt.show()

	plt.subplot(121)
	plt.hist(terdm, bins=100, range=(0, 3000), histtype='step', color='red')
	plt.twinx().plot(tE, ptEdu.sum(axis=1))
	plt.ylim(0)
	plt.subplot(122)
	plt.hist(durdm, bins=100, range=(0, 0.02), histtype='step', color='red')
	plt.twinx().plot(delta_u, np.nansum(ptEdu,axis=0))
	plt.ylim(0)
	#plt.hist2d(, bins=100, range=((0,1), (-1000,1000)))
	plt.show()

def parallax_cut(mass):
	PERIOD_EARTH = 365.2422
	alphaS = 80.8941667 * np.pi / 180.
	deltaS = -69.7561111 * np.pi / 180.
	epsilon = (90. - 66.56070833) * np.pi / 180.
	t_origin = 51442  # (21 septembre 1999) #58747 #(21 septembre 2019)

	sin_beta = np.cos(epsilon) * np.sin(deltaS) - np.sin(epsilon) * np.cos(deltaS) * np.sin(alphaS)
	beta = np.arcsin(sin_beta)  # ok because beta is in -pi/2; pi/2

	if abs(beta) == np.pi / 2:
		lambda_star = 0
	else:
		lambda_star = np.sign(
			(np.sin(epsilon) * np.sin(deltaS) + np.cos(epsilon) * np.sin(alphaS) * np.cos(deltaS)) / np.cos(
				beta)) * np.arccos(np.cos(deltaS) * np.cos(alphaS) / np.cos(beta))

	@nb.njit
	def parallax(t, mag, u0, t0, tE, delta_u, theta):
		out = np.zeros(t.shape)
		for i in range(len(t)):
			ti = t[i]
			tau = (ti - t0) / tE
			phi = 2 * np.pi * (ti - t_origin) / PERIOD_EARTH - lambda_star
			t1 = u0 ** 2 + tau ** 2
			t2 = delta_u ** 2 * (np.sin(phi) ** 2 + np.cos(phi) ** 2 * sin_beta ** 2)
			t3 = -2 * delta_u * u0 * (np.sin(phi) * np.sin(theta) + np.cos(phi) * np.cos(theta) * sin_beta)
			t4 = 2 * tau * delta_u * (np.sin(phi) * np.cos(theta) - np.cos(phi) * np.sin(theta) * sin_beta)
			u = np.sqrt(t1 + t2 + t3 + t4)
			out[i] = (u ** 2 + 2) / (u * np.sqrt(u ** 2 + 4))
		return out

	@nb.jit
	def microlens(t, mag, blend, u0, t0, tE, delta_u, theta):
		return - 2.5 * np.log10(blend * np.power(10, mag / -2.5) + (1 - blend) * np.power(10, mag / -2.5) * parallax(t, mag, u0, t0, tE, delta_u, theta))

	params = {
		'mag':19,
		'blend':0.,
		'u0':0.5,
		't0':(48928+52697)/2.,
		'tE':500,
		'delta_u':0.02,
		'theta':10.*np.pi/180.,
	}
	t = np.linspace(48928, 52697, 1000)
	cpara = microlens(t, **params)
	params['delta_u'] = 0
	cnopa = microlens(t, **params)
	plt.subplot(121)
	plt.plot(t, cpara)
	plt.plot(t, cnopa)
	#plt.gca().invert_yaxis()
	plt.subplot(122)
	plt.plot(t, np.abs(cpara-cnopa)/(cpara))
	plt.show()

test_generator(100, 1000000)