
import os
import pandas as pd
from IPython.display import clear_output

def update_progress(progress):
	bar_length = 40
	if isinstance(progress, int):
		progress = float(progress)
	if not isinstance(progress, float):
		progress = 0
	if progress < 0:
		progress = 0
	if progress >= 1:
		progress = 1

	block = int(round(bar_length * progress))

	clear_output(wait = True)
	text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
	print(text)

def load_results(dirpath):
	pds = []
	for (dirpath, dirnames, filenames) in os.walk(dirpath):
		for i, filename in enumerate(filenames):
			if filename[-4:]=='.pkl':
				pds.append(pd.read_pickle(os.path.join(dirpath,filename)))
			update_progress(i/len(filenames))
	return pd.concat(pds)
