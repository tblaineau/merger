from sklearn.neighbors import BallTree, KDTree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from itertools import combinations, chain
from scipy.special import comb

import sys
sys.path.append("/Users/tristanblaineau/Documents/Work/Python")
sys.path.append("/Users/tristanblaineau/Documents/Work/Python/merger/clean/libraries")
from lib_perso import *

def print_current(merged ,prefix=""):
	a1, d1 = merged[prefix+"am"].values, merged[prefix+"dm"].values
	a2, d2 = merged["ae"].values, merged["de"].values

	ax = plt.gca()
	segs = []
	widths=[]
	scale=3
	width_scale=1000

	a1 = np.array(a1)
	d1 = np.array(d1)
	a2 = np.array(a2)
	d2 = np.array(d2)
	for i, a in enumerate(a1):
		al = (a1[i]-a2[i])
		dl = (d1[i]-d2[i])
		segs.append(((a1[i]+scale*al, d1[i]+scale*dl), (a2[i]-scale*al, d2[i]-scale*dl)))
		widths.append(np.sqrt(al*al+dl*dl)*width_scale)
	ln_coll = LineCollection(segs, color="black", linewidth=widths)
	ln_coll.set_antialiased(True)
	ax.add_collection(ln_coll)

	ax.axis("equal")
	ax.invert_xaxis()
	plt.legend()
	#plt.xlim(82.6, 81.9)
	#plt.ylim(-71.4, -71)
	#plt.show()

def fusion(stars1, stars2, prefix=''):
	tree1 = KDTree(stars1[[prefix+"am",prefix+"dm"]], leaf_size=50)
	tree2 = KDTree(stars2[["ae","de"]], leaf_size=50)

	dist1, ind1 = tree1.query(stars2[["ae","de"]], k=3, return_distance=True, dualtree=True)
	dist2, ind2 = tree2.query(stars1[[prefix+"am",prefix+"dm"]], k=3, return_distance=True, dualtree=True)

	ind1 = pd.DataFrame(ind1, columns=['c1', 'c2', 'c3'])
	ind2 = pd.DataFrame(ind2, columns=['c1', 'c2', 'c3'])

	print("merge !")
	m_stars2 = ind1.join(stars2)
	m_stars1 = ind2.join(stars1)

	mask = m_stars2.index.values == m_stars1.loc[m_stars2.c1].c1.values
	return pd.merge(m_stars2[mask], m_stars1, right_index=True, left_on="c1"), np.median(dist1[mask])

def correction(merged, macho_stars):
	"""Corrects macho_stars positions from merged
	
	[description]
	
	Arguments:
		merged {dataframe} -- Contains associated stars
		macho_stars {dataframe} -- contains original macho stars
	
	Returns:
		[dataframe] -- macho_stars with updated positions
	"""
	merged["ac"]=merged["ae"]-merged["am"]
	merged["dc"]=merged["de"]-merged["dm"]
	corr = merged.groupby(["chunk","template_pier"])[["ac", "dc"]].agg("mean")
	new_macho_stars = pd.merge(macho_stars, corr, left_on=["chunk", "template_pier"], right_index=True)
	new_macho_stars["am"] = new_macho_stars["ac"]+new_macho_stars["am"]
	new_macho_stars["dm"] = new_macho_stars["dc"]+new_macho_stars["dm"]
	new_macho_stars.dropna(subset=["am", "dm"], how='any', inplace=True)
	return new_macho_stars.reset_index()

def compute_distances(stars, dlist=[], i=1):
	for idx, s in enumerate(stars[i:]):
		dlist.append([i-1, i+idx, (stars[i-1][1]-s[1])**2 + (stars[i-1][2]-s[2])**2])
	if len(stars)==i+1:
		return 0
	else:
		compute_distances(stars, dlist, i+1)

def most_distant(stars):
	dlist = []
	compute_distances(stars, dlist)
	dlist=np.array(dlist)
	dlist = dlist[dlist[:, 2].argsort()[::-1]]
	return dlist[0]

def comb_index(n, k):
	#generate indices of combinations without repititon of a n*k array
    count = comb(n, k, exact=True)
    index = np.fromiter(chain.from_iterable(combinations(range(n), k)), 
                        int, count=count*k)
    return index.reshape(-1, k)

def generate_quads(ss):
	quads=[]
	for idx1, star1 in enumerate(ss[:-3]):
		for idx2, star2 in enumerate(ss[idx1+1:-2]):
			for idx3, star3 in enumerate(ss[idx1+idx2+2:-1]):
				for idx4, star4 in enumerate(ss[idx1+idx2+idx3+3:]):
					stars = np.array([star1, star2, star3, star4])
					dlist = most_distant(stars)
					choice = stars[dlist[[0,1]].astype(int)]
					if choice[0,1]<choice[1,1]:
						starA = choice[0]
						starB = choice[1]
					else:
						starA = choice[1]
						starB = choice[0]

					Xb, Yb = starB[1], starB[2]
					Xa, Ya = starA[1], starA[2]
					starC, starD = np.delete(stars, dlist[[0,1]].astype(int), axis=0)
					xc = (starC[1]-Xa)/(Xb-Xa)
					yc = (starC[2]-Ya)/(Yb-Ya)
					xd = (starD[1]-Xa)/(Xb-Xa)
					yd = (starD[2]-Ya)/(Yb-Ya)
					if xc>xd:
						xc,yc,xd,yd = xd,yd,xc,yc
						starC, starD = starD, starC
					distAB = np.sqrt((Xa-Xb)**2+(Ya-Yb)**2)
					quads.append([xc, yc, xd, yd, starA[0], starB[0], starC[0], starD[0], distAB])
	print(len(quads))
	return np.array(quads)

def vectorized_generate_quads(ss):
	id_M, am, dm = ss.dtype.names
	idx = comb_index(ss.shape[0], 4)
	idx_dist = comb_index(4, 2)
	stars = ss[idx]
	stars_idxs = np.arange(stars.shape[0])
	pairs = stars[:,idx_dist]
	mindist_idx = ((pairs[:,:,0][am] - pairs[:,:,1][am])**2+(pairs[:,:,0][dm] - pairs[:,:,1][dm])**2).argmin(axis=1)
	a = pairs[stars_idxs, mindist_idx]
	c = a[:,0][am]<a[:,1][dm]
	ABpairs_idx = idx_dist[mindist_idx]
	CDpairs_idx = idx_dist[~mindist_idx]
	A_idx = np.where(c, ABpairs_idx[:,0], ABpairs_idx[:,1])
	B_idx = np.where(~c, ABpairs_idx[:,0], ABpairs_idx[:,1])
	A = stars[stars_idxs, A_idx]
	B = stars[stars_idxs, B_idx]
	C = stars[stars_idxs, CDpairs_idx[:,0]]
	D = stars[stars_idxs, CDpairs_idx[:,1]]
	xc = (C[am] - A[am])/(B[am] - A[am])
	yc = (C[dm] - A[dm])/(B[dm] - A[dm])
	xd = (D[am] - A[am])/(B[am] - A[am])
	yd = (D[dm] - A[dm])/(B[dm] - A[dm])
	c = xc>xd
	#==============================
	#LENT
	xc[c], xd[c] = xd[c], xc[c]
	yc[c], yd[c] = yd[c], yc[c]
	C[c], D[c] = D[c], C[c]
	#==============================
	distAB = np.sqrt((A[am]-B[am])**2+(A[dm]-B[dm])**2)
	return np.array([xc, yc, xd, yd, A[id_M], B[id_M], C[id_M], D[id_M], distAB]).T


def quads(subdf, eros_stars, macho_stars, nb_stars=10):
	print(str(subdf.iloc[0].chunk)+" "+subdf.iloc[0].template_pier)
	if len(subdf)<100 or subdf.iloc[0].chunk==255:
		return subdf.drop(['ae', 'alpha_E', 'blue_E', 'c1', 'c1_x', 'c1_y', 'c2_x', 'c2_y', 'c3_x', 'c3_y', 'de', 'delta_E', 'id_E', 'red_E'], axis=1)
	#select 10 stars from each
	
	tree1 = KDTree(subdf[["am","dm"]], leaf_size=50)
	tree2 = KDTree(eros_stars[["ae","de"]], leaf_size=50)

	dist1, ind1 = tree1.query(eros_stars[["ae","de"]], k=3, return_distance=True, dualtree=True)
	dist2, ind2 = tree2.query(subdf[["am","dm"]], k=3, return_distance=True, dualtree=True)

	ind1 = pd.DataFrame(ind1, columns=['c1', 'c2', 'c3'])
	ind2 = pd.DataFrame(ind2, columns=['c1', 'c2', 'c3'])

	mask = ind2.index.values == ind1.loc[ind2.c1].c1.values
	# print(dist2)
	curr_mean_dist = np.mean(dist2[mask][:,0])
	# curr_mean_dist = 0.0005

	# ON MAGNITUDE
	es = subdf.sort_values(["red_E", "blue_E"]).iloc[:nb_stars][["id_E", "ae", "de"]].to_records(index=False)
	ms = subdf.sort_values(["red_M", "blue_M"]).iloc[:nb_stars][["id_M", "am", "dm"]].to_records(index=False)


	#print(subdf.sort_values(["red_E", "blue_E"])["red_E"])
	#print(subdf.sort_values(["red_M", "blue_M"])["red_M"])

	# INRADIUS
	# selected = subdf[["id_M", "am", "dm"]].sample(n=1).values
	# print(selected)
	# radius = 10
	# es = KDTree(eros_stars[['ae', 'de']]).query_radius([selected[0, [1, 2]]], r = radius*2.7e-4)[0]
	# ms = KDTree(macho_stars[['am', 'dm']]).query_radius([selected[0, [1, 2]]], r = radius*2.7e-4)[0]

	# es = eros_stars.loc[es][["id_E", "ae", "de"]].values
	# ms = macho_stars.loc[ms][["id_M", "am", "dm"]].values

	#RANDOM [["id_M", "am", "dm"]] [["id_E", "ae", "de"]]
	# ms = subdf.sample(n=nb_stars)
	# es = np.unique(pd.concat([eros_stars.loc[ms["c1_y"]], eros_stars.loc[ms["c2_y"]], eros_stars.loc[ms["c3_y"]]])[["id_E", "ae", "de"]].dropna(how='any').to_records(index=False))
	# ms = ms[["id_M", "am", "dm"]].dropna(how='any').to_records(index=False)

	#MIXED
	# es1 = subdf.sort_values(["red_E", "blue_E"]).iloc[:int(nb_stars/2)][["id_E", "ae", "de"]].to_records(index=False)
	# ms1 = subdf.sort_values(["red_M", "blue_M"]).iloc[:int(nb_stars/2)][["id_M", "am", "dm"]].to_records(index=False)
	# ms = subdf.sample(n=int(nb_stars/2))
	# es = np.unique(np.hstack((es1, pd.concat([eros_stars.loc[ms["c1_y"]], eros_stars.loc[ms["c2_y"]], eros_stars.loc[ms["c3_y"]]])[["id_E", "ae", "de"]].dropna(how='any').to_records(index=False))))
	# ms = np.unique(np.hstack((ms1, ms[["id_M", "am", "dm"]].dropna(how='any').to_records(index=False))))

	# plt.scatter(es[:,1], es[:,2])
	# plt.scatter(ms[:,1], ms[:,2])
	# plt.scatter(subdf["ae"], subdf["de"], s=1)
	# plt.scatter(subdf["am"], subdf["dm"], s=1)
	# plt.gca().axis('equal')
	# plt.show()
	

	print(len(es), len(ms))
	print(ms.dtype)

	m_quads = vectorized_generate_quads(ms)
	e_quads = vectorized_generate_quads(es)

	tree = KDTree(m_quads[:,0:4])
	dist, ind = tree.query(e_quads[:,0:4], 3)
	distind = np.concatenate((dist, ind), axis=1)
	distind = pd.DataFrame(distind)
	distind.columns = ["d1", 'd2', 'd3', 'i1', 'i2', 'i3'] #<- ix index MACHO, row = EROS idx
	distind = distind.sort_values(['d1', 'd2', 'd3'])

	#translation
	SCALE_LIMIT=1.1
	neros_stars = eros_stars.set_index('id_E')
	nmacho_stars = subdf.set_index('id_M')

	# e_quads_tmp = pd.DataFrame(e_quads, columns=['xc', 'yc', 'xd', 'yd', 'starA', 'starB', 'starC', 'starD', 'distAB'])
	# e_quads_tmp.sort_values('distAB', inplace=True, ascending=False)#, kind='mergesort')
	# print(e_quads_tmp.starA)
	# print(distind)
	# distind = pd.concat((distind.reindex(e_quads_tmp.index), e_quads_tmp.distAB.rename('distAB')),axis=1)
	# distind = distind.reset_index().groupby('distAB').apply(lambda x:x.sort_values('d1').iloc[0])
	# distind = distind.set_index('index').sort_values('distAB', ascending=False)
	print(distind)

	bot = []

	counter=0
	for row in distind.itertuples(name='star'):
		if counter>=10:
			break
		idx = row.Index
		print(idx)
		q1 = e_quads[idx]
		q2 = m_quads[int(row.i1)]
		seA = neros_stars.loc[q1[4]].to_dict()
		seB = neros_stars.loc[q1[5]].to_dict()
		smA = nmacho_stars.loc[q2[4]].to_dict()
		smB = nmacho_stars.loc[q2[5]].to_dict()

		v1 = [seB['ae']-seA['ae'], seB['de']-seA['de']]
		v1prime = [smB['am']-smA['am'], smB['dm']-smA['dm']]
		scale = np.linalg.norm(v1)/np.linalg.norm(v1prime)

		if SCALE_LIMIT > scale > 1/SCALE_LIMIT:
			counter+=1
			at = seA['ae']-smA['am']
			dt = seA['de']-smA['dm']
			if True:#abs(at)<5*2.7e-4 and abs(dt) <5*2.7e-4:
				t_am = subdf.loc[:,"am"]+at
				t_dm = subdf.loc[:,"dm"]+dt

				# for idx in q1:
				# 	tmp = subdf[eros_stars.id_E==idx]
				# 	plt.scatter(tmp.t_am, tmp.t_dm, marker='x', color='green')

				t_am = seA['ae'] + (t_am-seA['ae'])*scale
				t_dm = seA['de'] + (t_dm-seA['de'])*scale

				# subdf.loc[:,"s_am"] = seA['ae'] + (t_am-seA['ae'])*scale
				# subdf.loc[:,"s_dm"] = seA['de'] + (t_dm-seA['de'])*scale

				theta = np.arccos(np.dot(v1, v1prime)/(np.linalg.norm(v1)*np.linalg.norm(v1prime)))
				cross = np.cross(v1, v1prime)
				print(cross)
				plane = np.cross([1,0], [0, 1])
				if np.dot(plane, cross)<0:
					theta = -theta
				print(str(theta)+" <- theta")
				print(str(scale)+" <- scale")

				t_am = (t_am-seA['ae'])*np.cos(theta) + (t_dm-seA['de'])*np.sin(theta) + seA['ae']
				t_dm = -(t_am-seA['ae'])*np.sin(theta) + (t_dm-seA['de'])*np.cos(theta) + seA['de']


				tree1 = KDTree(np.array([t_am, t_dm]).T, leaf_size=50)
				tree2 = KDTree(eros_stars[["ae","de"]], leaf_size=50)

				dist1, ind1 = tree1.query(eros_stars[["ae","de"]], k=3, return_distance=True, dualtree=True)
				dist2, ind2 = tree2.query(np.array([t_am, t_dm]).T, k=3, return_distance=True, dualtree=True)

				ind1 = pd.DataFrame(ind1, columns=['c1', 'c2', 'c3'])
				ind2 = pd.DataFrame(ind2, columns=['c1', 'c2', 'c3'])

				mask1 = ind2.index.values == ind1.loc[ind2.c1].c1.values

				# print(str(np.percentile(dist2[mask1][:,0], 0.99))+"<- new dist")
				print(str(dist2[mask1][:,0].mean())+"<- new dist")
				print(str(curr_mean_dist)+"<- current_dist")
				# i=0
				# plt.scatter(subdf["am"], subdf["dm"], s=10, marker='o',color='gray')
				# plt.scatter(es[:,1], es[:,2], marker='o', color='purple')
				# plt.scatter(ms[:,1], ms[:,2], marker='o', color='pink')
				# for idx in q2[[4,5,6,7]]:
				# 	i+=1
				# 	tmp = macho_stars[macho_stars.id_M==idx]
				# 	if len(tmp)>0:
				# 		plt.scatter(tmp.am, tmp.dm, marker='+', color='pink')
				# 		print(tmp.am.values)
				# 		plt.text(tmp.am.values, tmp.dm.values, s=str(i))
				# plt.scatter(seA['ae'], seA['de'], marker='o', color='black')
				# plt.scatter(seB['ae'], seB['de'], marker='o', color='black')
				# for idx in q1:
				# 	tmp = eros_stars[eros_stars.id_E==idx]
				# 	plt.scatter(tmp.ae, tmp.de, marker='+', color='blue')
				# # for idx in q2:
				# # 	tmp = subdf[subdf.id_M==idx]
				# # 	plt.scatter(tmp.s_am, tmp.s_dm, marker='x', color='darkred')
				# for idx in q2:
				# 	tmp = subdf[subdf.id_M==idx]
				# 	plt.scatter(tmp.t_am, tmp.t_dm, marker='+', color='red')

				# fus, med = fusion(subdf.reset_index(drop=True).drop(['c1', 'c1_x', 'c2_x', 'c3_x', 'id_E', 'alpha_E', 'delta_E', 'red_E', 'blue_E', 'ae', 'de', 'c1_y', 'c2_y', 'c3_y'], axis=1), eros_stars, prefix='t_')
				# # print_current(fus, 't_')
				# plt.scatter(fus["ae"], fus["de"], s=1)
				# plt.scatter(fus["t_am"], fus["t_dm"], s=1)
				# plt.gca().axis('equal')
				# print(str(med)+" <- returned median")
				# plt.show()

				if curr_mean_dist==0:
					curr_mean_dist = np.mean(dist2[mask1][:,0])
				#bot.append((idx,np.percentile(dist2[mask1][:,0], 0.95)))
				bot.append((idx,dist2[mask1][:,0].mean()))

	print(bot)
	bot = np.array(bot, dtype=[('idx', float), ('value', float)])
	print(bot)
	output_idx = np.sort(bot, order='value')[0][0]
	print(output_idx)

	q1 = e_quads[int(output_idx)]
	q2 = m_quads[int(distind.loc[output_idx].i1)]
	seA = neros_stars.loc[q1[4]].to_dict()
	seB = neros_stars.loc[q1[5]].to_dict()
	smA = nmacho_stars.loc[q2[4]].to_dict()
	smB = nmacho_stars.loc[q2[5]].to_dict()

	v1 = [seB['ae']-seA['ae'], seB['de']-seA['de']]
	v1prime = [smB['am']-smA['am'], smB['dm']-smA['dm']]
	scale = np.linalg.norm(v1)/np.linalg.norm(v1prime)

	if SCALE_LIMIT > scale > 1/SCALE_LIMIT:
		at = seA['ae']-smA['am']
		dt = seA['de']-smA['dm']
		subdf.loc[:,"t_am"] = subdf.loc[:,"am"]+at
		subdf.loc[:,"t_dm"] = subdf.loc[:,"dm"]+dt
		subdf.loc[:,"t_am"] = seA['ae'] + (subdf.loc[:,"t_am"]-seA['ae'])*scale
		subdf.loc[:,"t_dm"] = seA['de'] + (subdf.loc[:,"t_dm"]-seA['de'])*scale
		theta = np.arccos(np.dot(v1, v1prime)/(np.linalg.norm(v1)*np.linalg.norm(v1prime)))
		plane = np.cross([1,0], [0, 1])
		if (np.dot(plane, cross)<0):
			theta = -theta
		subdf.loc[:,"rot_am"] = (subdf.loc[:,"t_am"]-seA['ae'])*np.cos(theta) + (subdf.loc[:,"t_dm"]-seA['de'])*np.sin(theta) + seA['ae']
		subdf.loc[:,"rot_dm"] = -(subdf.loc[:,"t_am"]-seA['ae'])*np.sin(theta) + (subdf.loc[:,"t_dm"]-seA['de'])*np.cos(theta) + seA['de']

		subdf.loc[:,"am"] = subdf.loc[:,"rot_am"]
		subdf.loc[:,"dm"] = subdf.loc[:,"rot_dm"]
		return subdf.drop(['c1', 'c1_x', 'c2_x', 'c3_x', 'id_E', 'alpha_E', 'delta_E', 'red_E', 'blue_E', 'ae', 'de', 'c1_y', 'c2_y', 'c3_y','t_am', 't_dm', 'rot_am', 'rot_dm'], axis=1)


	print("NO VALID QUADS !!!!")
	return subdf.drop(['ae', 'alpha_E', 'blue_E', 'c1', 'c1_x', 'c1_y', 'c2_x', 'c2_y', 'c3_x', 'c3_y', 'de', 'delta_E', 'id_E', 'red_E'], axis=1)

eros_stars = pd.DataFrame(load_eros_field_stars("032"))
eros_stars.columns = ["id_E", "alpha_E", "delta_E", "red_E", "blue_E"]
eros_stars = eros_stars.astype({"id_E":object, "alpha_E":float, "delta_E":float, "red_E":float, "blue_E":float}, copy=False)
eros_stars.loc[:,"alpha_E"] = eros_stars["alpha_E"]*np.pi/180.
eros_stars.loc[:,"delta_E"] = eros_stars["delta_E"]*np.pi/180.
print('EROS stars loaded.')

macho_stars = pd.read_pickle("macho_mags49.pkl")
# macho_stars = pd.DataFrame(load_macho_field_stars(49))
# macho_stars.columns = ["id_M", "alpha_M", "delta_M", "template_pier", "chunk"]
# macho_stars = macho_stars.astype({"id_M":object, "alpha_M":float, "delta_M":float, "template_pier":object, "chunk":float})
# macho_mags = merger_library.load_macho_field("/Volumes/DisqueSauvegarde/MACHO/lightcurves/", 49)
# macho_mags = macho_mags.replace(to_replace=[99.999,-99.], value=np.nan).dropna(axis=0, how='all', subset=['blue_M', 'red_M']).groupby('id_M')[['red_M', 'blue_M']].agg('mean')
# macho_stars = pd.merge(macho_stars, macho_mags, left_on='id_M', right_index=True)
# macho_stars.to_pickle("macho_mags49.pkl")
print('MACHO stars loaded.')

eros_stars.loc[:,"ae"], eros_stars.loc[:,"de"] = proj_ad(eros_stars["alpha_E"], eros_stars["delta_E"])
macho_stars.loc[:,"am"], macho_stars.loc[:,"dm"] = proj_ad(macho_stars["alpha_M"], macho_stars["delta_M"])

print("GO!")

merged, mean_dist = fusion(macho_stars, eros_stars)
print("QUADS")
new_macho_stars = merged.groupby(["template_pier", "chunk"]).apply(quads, eros_stars=eros_stars, macho_stars=macho_stars, nb_stars=80)	#[(merged.chunk==31) & (merged.template_pier == 'E')]

pd.to_pickle(new_macho_stars, 'correct1.pkl')
new_macho_stars = pd.read_pickle('correct1.pkl')


print_current(merged)
new_macho_stars = new_macho_stars[new_macho_stars.chunk!=255].reset_index(drop=True)
merged, mean_dist = fusion(new_macho_stars, eros_stars)
plt.figure()
print_current(merged)
plt.show()

# for i in range(1):
# 	print(i)
# 	print("Correction")
# 	new_macho_stars = correction(merged, macho_stars)
# 	print("Merge")
# 	merged, _ = fusion(new_macho_stars, eros_stars)

# print_current(merged)
# plt.show()