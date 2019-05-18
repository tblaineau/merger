import numpy as np
import numba as nb

@nb.njit
def randtobest1(candidate, best, samples, dim=3, scale=1.):
	out = np.zeros(dim)
	for i in range(dim):
		out[i] = candidate[i] + scale*(best[i] - candidate[i] + samples[0][i] - samples[1][i])
	return out

