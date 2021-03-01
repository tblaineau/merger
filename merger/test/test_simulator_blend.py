import merger.hst_blending.simulation_blending_work as sbw
import pandas as pd
import numpy as np
from merger.clean.libraries.merger_library import COLOR_FILTERS
import matplotlib.pyplot as plt


df = pd.read_pickle("42_1.bz2", compression='bz2')
fraction=0.5
df = df.groupby(["id_E", "id_M"]).filter(lambda x: np.random.random()<fraction)

print("error-magnitude")
er = sbw.ErrorMagnitudeRelation(df, list(COLOR_FILTERS.keys()), bin_number=20)

print("generator")
infos = df.groupby(["id_E", "id_M"])[list(COLOR_FILTERS.keys())].agg("median")
t0_ranges = df.groupby(["id_E", "id_M"])["time"].agg(["min", "max"]).values.T
print(t0_ranges)
rg = sbw.RealisticGenerator(infos.index.get_level_values(0).values, infos.blue_E.values, blend_directory="/Users/tristanblaineau/Documents/Work/Jupyter/blend/HST/HST_FINAL", xvt_file="xvts_clean.npy", densities_path="../clean/useful_files/densities.txt")

pms = rg.generate_parameters(nb_parameters=len(infos), t0_ranges=t0_ranges)
print(pms)

plt.hist(t0_ranges.flatten(), bins=100)
plt.hist(pms["t0"], bins=100)
plt.show()