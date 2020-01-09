import numpy as np
import numba as nb

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


fastmath = False


@nb.njit(fastmath=fastmath)
def microlens_simple_FAST(t, mag, blend, u0, t0, tE, delta_u=0, theta=0):
	return microlens_simple(t, mag, blend, u0, t0, tE, delta_u=0, theta=0)


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
	bounds : sequence
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