import pandas as pd
import os
from itertools import chain

fs = []
for (root, dir, filenames) in chain.from_iterable(os.walk(path) for path in ["./int6M01", "./int6M12"]):
	for file in filenames:
		print(file[-3:])
		if file[-3:] == "pkl":
			fs.append(pd.read_pickle(os.path.join(root, file)))

all = pd.concat(fs)
all.reset_index(inplace=True)
all.set_index('idx', inplace=True)
all = all[all.index.value_counts()==2]
pd.to_pickle(all, 'integral6Mbig.pkl')