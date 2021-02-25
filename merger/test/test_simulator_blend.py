import merger.hst_blending.simulation_blending_work as sbw


er = sbw.ErrorMagnitudeRelation()
rg = sbw.RealisticGenerator()

df = pd.read_pickle("42_1.bz2", compression='bz2')
rg.generate_parameters()