import numpy as np
import glob
import os

path_to_pairs = "prod_pairs.txt"
path_to_merged = "path"

if path_to_merged:
	pp = []
	pp = list(glob.glob(os.path.join(path_to_merged, "F_*", "*.bz2")))
#elif path_to_pairs:
#	pp = open("path_to_pairs", "r").readlines()


chosen_pairs = np.random.choice(pp, size=10, replace=False)

