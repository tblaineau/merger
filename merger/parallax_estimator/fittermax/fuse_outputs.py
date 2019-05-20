import pandas as pd
import os

fs = []
for (_, _, filenames) in os.walk("."):
	for file in filenames:
		print(file[-3:])
		if file[-3:] == "pkl":
			fs.append(pd.read_pickle(file))

all = pd.concat(fs)
pd.to_pickle(all, 'fastscipyminmax6M.pkl')