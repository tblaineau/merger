import pandas as pd
import os

fs = []
for (root, dir, filenames) in os.walk("./6M12"):
	for file in filenames:
		print(file[-3:])
		if file[-3:] == "pkl":
			fs.append(pd.read_pickle(os.path.join(root, file)))

all = pd.concat(fs)
all.reset_index(inplace=True)
pd.to_pickle(all, 'fastscipyminmax6M12.pkl')