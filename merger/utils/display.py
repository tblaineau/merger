import matplotlib.pyplot as plt
import logging

def hist2d1d(x, y, bins=10, figsize=None, hist_args={"histtype":"step"}, **kwargs):
	fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize, gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [1, 3]})
	ax[1, 0].hist2d(x, y, bins=bins, **kwargs)
	try:
		if not isinstance(bins, int):
			binx = bins[0]
			biny = bins[1]
		else:
			binx = bins
			biny = bins
	except ValueError:
		logging.error(f"Bad value for bins, expected int or sequence(int, int) : {bins}")
	ax[0, 0].hist(x, bins=binx, **hist_args)
	ax[1, 1].hist(y, bins=biny, orientation='horizontal', **hist_args)
	ax[0, 1].axis('off')
