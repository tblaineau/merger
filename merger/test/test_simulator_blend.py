import merger.hst_blending.simulation_blending_work as sbw
import pandas as pd
import numpy as np
from merger.clean.libraries.merger_library import COLOR_FILTERS


df = pd.read_pickle("42_1.bz2", compression='bz2')
fraction=0.5
df = df.groupby(["id_E", "id_M"]).filter(lambda x: np.random.random()<fraction)

print("error-magnitude")
er = sbw.ErrorMagnitudeRelation(df, list(COLOR_FILTERS.keys()), bin_number=20)

print("generator")
infos = df.groupby(["id_E", "id_M"])[list(COLOR_FILTERS.keys())].agg("median")
print()
rg = sbw.RealisticGenerator(infos.index.get_level_values(0).values, infos.blue_E.values)


rg.generate_parameters()