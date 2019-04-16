import numpy as np
import astropy.units as u
import astropy.constants as constants

import matplotlib.pyplot as plt

class Microlensing_parameters_generator():
	def __init__(self, mass, u_lim, v0):
		self.mass = mass<<u.Msun
		self.u_lim = u_lim
		self.v0 = v0 << (u.km/u.s)

		self.t_min = 48928 << u.d
		self.t_max = 52697 << u.d
		self.t_obs = self.t_max - self.t_min

		self.r_lmc = 55 << u.kpc
		self.rho_0 = 0.0079 << (u.Msun/u.pc**3)

		d_sol = 8.5 << u.kpc
		a = 5 << u.kpc
		l_lmc, b_lmc = 280.4652 / 180. * np.pi, -32.8884 / 180. * np.pi
		cosb_lmc = np.cos(b_lmc)
		cosl_lmc = np.cos(l_lmc)

		self.A = d_sol ** 2 + a ** 2
		self.B = d_sol * cosb_lmc * cosl_lmc

		self.r0 = np.sqrt(4 * constants.G / (constants.c ** 2) * self.r_lmc)

	def R_E(self, x):
		return self.r0 * np.sqrt(self.mass * x * (1-x))

	def rho_x(self, x):
		return self.rho_0 * self.A / ((x * self.r_lmc) ** 2 - 2 * x * self.r_lmc * self.B + self.A) / self.mass

	def p_xvT(self, x, v_T):
		q = self.rho_x(x) / self.mass * self.r_lmc * (2*self.u_lim*self.R_E(x)*self.t_obs * abs(v_T<<(u.km/u.s)))
		return q.decompose().value

	def f_x(self, x):
		return self.rho_x(x) * x**2

	def f_vT(self, v_T):
		return (2*v_T/(self.v0**2)) * np.exp(-v_T**2/(self.v0**2))

g = Microlensing_parameters_generator(mass=100, u_lim=1, v0=220)

x = np.linspace(0,1,100)
v_T = np.linspace(-600, 600, 100)
res = g.p_xvT(x[:,None], v_T[None,:])
plt.imshow(res)
plt.show()