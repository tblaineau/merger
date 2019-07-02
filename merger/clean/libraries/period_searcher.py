import numba as nb
import numpy as np
import scipy.signal
import scipy.special


@nb.njit
def numba_mean(values):
	s = 0
	for i in range(len(values)):
		s+=values[i]
	return s/len(values)


@nb.njit
def chi2_int(phase, mag, err):
	chi2 = 0
	lt = len(phase)
	for i in range(0, lt - 2):
		if (phase[i + 2] - phase[i]) == 0:
			ri = 0.5
		else:
			ri = (phase[i + 1] - phase[i]) / (phase[i + 2] - phase[i])
		sigmaisq = err[i + 1] ** 2 + (1 - ri) ** 2 * err[i] ** 2 + ri ** 2 * err[i + 2] ** 2
		chi2 += (mag[i + 1] - mag[i] - ri * (mag[i + 2] - mag[i])) ** 2 / sigmaisq

	ri = (phase[lt - 1] - phase[lt - 2]) / (phase[0] - phase[lt - 2])
	sigmaisq = err[lt - 1] ** 2 + (1 - ri) ** 2 * err[lt - 2] ** 2 + ri ** 2 * err[0] ** 2
	chi2 += (mag[lt - 1] - mag[lt - 2] - ri * (mag[0] - mag[lt - 2])) ** 2 / sigmaisq
	return chi2


@nb.njit
def period_search_loop(time, mag, err, frequency):
	phase=[]
	for i in range(len(time)):
		phase.append(time[i]%(1/frequency))
	phase = np.array(phase)
	idx = np.argsort(phase)
	return chi2_int(phase[idx], mag[idx], err[idx])


@nb.njit
def auto_correlation(time, mag, err, min_freq, step_freq, nb_steps):
	chi2s = []
	freqs = []
	for i in range(nb_steps):
		chi2s.append(period_search_loop(time, mag, err, min_freq+i*step_freq))
		freqs.append(min_freq+i*step_freq)
	return chi2s, freqs


def confidence(time, mag, err, min_freq, step_freq, nb_steps):
	"""
	Parameters
	----------
	time : list-like

	mag : array-like

	err : array-like

	min_freq : float
		starting frequency to explore
	step_freq : float
		frequency step, ideally should be 1/10T
	nb_steps : int
		numbers of step to compute

	Returns
	-------

	np.ndarray
		list of the reduced chi2s
	np.ndarray
		list of frequencies computed
	float
		frequency of the minimum chi2
	float
		probability of the minimum chi2 found
	float
		minimum chi2 value
	"""
	chi2s, freqs = auto_correlation(time, mag, err, min_freq, step_freq, nb_steps)
	freqs = np.array(freqs)
	mean_chi2 = np.mean(chi2s)
	reduced_chi2s = ((chi2s - mean_chi2) / mean_chi2) * np.sqrt(len(time) / 2.)
	min_chi2_idx = reduced_chi2s.argmin()
	min_chi2 = reduced_chi2s[min_chi2_idx]
	if mean_chi2 <= 0:
		best_proba = 1.
	else:
		best_proba = np.log10(nb_steps / 2. * scipy.special.erfc(-min_chi2 / np.sqrt(2)))
	return reduced_chi2s, freqs, best_proba, freqs[min_chi2_idx], min_chi2, freqs[0], np.log10(nb_steps / 2. * scipy.special.erfc(-reduced_chi2s[0] / np.sqrt(2)))

def confidence_use(time, mag, err, nb_steps=1500):
	time_legnth = time.max() - time.min()
	min_freq = 1 / time_legnth
	freq_step = 1 / (10 * time_legnth)
	_, _, best_proba, best_freq, _ = confidence(time, mag, err, min_freq, freq_step, nb_steps)
	return best_proba, best_freq, min_freq