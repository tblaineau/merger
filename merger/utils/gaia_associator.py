import numpy as np
import numba as nb
import sys
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import scipy.optimize
from sklearn.neighbors import KDTree
import os

def right_ascension_to_radians(ra):
	if isinstance(ra, str):
		ra = ra.split(":")
	return 2.*np.pi/24.*(float(ra[0]) + (float(ra[1]) + float(ra[2])/60.)/60.)

def declination_to_radians(dec):
	if isinstance(dec, str):
		dec = dec.split(":")
	return 2.*np.pi/360.*(abs(float(dec[0])) + (float(dec[1]) + float(dec[2])/60.)/60.) * np.sign(float(dec[0]))


@nb.jit
def rotation_sphere(ra, dec, ra0, dec0, theta):
	cosra = np.cos(ra0)
	sinra = np.sin(ra0)
	cosdec = np.cos(dec0)
	sindec = np.sin(dec0)

	cosra_d = np.cos(ra)
	sinra_d = np.sin(ra)
	cosdec_d = np.cos(dec)
	sindec_d = np.sin(dec)

	K = np.array([[0, -sindec, cosdec * sinra],
				  [sindec, 0, -cosdec * cosra],
				  [-cosdec * sinra, cosdec * cosra, 0]
				  ])
	#k = np.array([cosdec * cosra, cosdec * sinra, sindec])
	v = np.array([cosdec_d * cosra_d, cosdec_d * sinra_d, sindec_d])
	R = np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
	vrot = R @ v
	dec = np.arctan2(vrot[2], np.sqrt(vrot[0] ** 2 + vrot[1] ** 2))
	ra = np.arctan2(vrot[1] / np.cos(dec), vrot[0] / np.cos(dec))
	return np.array([ra, dec]).T


@nb.jit
def transform(ra, dec, ra0, dec0, r, a, alpha, theta):
	out = rotation_sphere(ra, dec, ra0, dec0, theta)
	ra1, dec1 = out[:, 0], out[:, 1]
	sina = np.sin(alpha)
	cosa = np.cos(alpha)
	a1 = r * (1 + (a - 1)*cosa**2)
	a2 = r * (a - 1) * cosa * sina
	b2 = r * (1 + (a - 1)*sina**2)
	ra_p = ra0 + a1 * (ra1 - ra0) + a2 * (dec1 - dec0) #+ offra
	dec_p = dec0 + a2 * (ra1 - ra0) + b2 * (dec1 - dec0) #+ offdec
	return np.array([ra_p, dec_p]).T


@nb.njit
def to_cartesian(o):
	s=[]
	for i in range(len(o)):
		cos_dec = np.cos(o[i,1])
		s.append([cos_dec*np.cos(o[i,0]), cos_dec*np.sin(o[i,0]), np.sin(o[i,1])])
	return s

field =  sys.argv[1]
MACHO_path = "/pbs/home/b/blaineau/data/MACHO"
#MACHO_path = "/Volumes/DisqueSauvegarde/MACHO/star_coordinates"
MACHO = os.path.join(MACHO_path, "DumpStar_"+str(field)+".txt")

macho = []
with open(MACHO) as f:
	for l in f.readlines():
		l  =l.split(";")
		macho.append([l[3], l[4], l[6], l[7], l[8], ":".join(l[0:3])])
macho = np.array(macho)

macho_rad = []
for i in macho:
	macho_rad.append([right_ascension_to_radians(i[0]), declination_to_radians(i[1])])
macho_rad = np.array(macho_rad)
macho_coord = SkyCoord(macho_rad[:,0], macho_rad[:,1], unit=u.rad)

print("Loading Gaia")
gaia_path = "/pbs/home/b/blaineau/work/association"
#gaia_path = "/Users/tristanblaineau/Documents/Work/Jupyter/quad_merge"
gaia = pd.read_feather(os.path.join(gaia_path, "lmcgaiafull.feather"))
print("Done")
gaia_rad = np.array([gaia.ra_epoch2000.values*np.pi/180, gaia.dec_epoch2000.values*np.pi/180]).T
gaia_rad = np.append(gaia_rad, np.arange(0, len(gaia_rad)).reshape(len(gaia_rad), 1), axis=1)
gaia_coord = SkyCoord(gaia.ra_epoch2000.values, gaia.dec_epoch2000.values, unit=u.deg)

center = SkyCoord(np.median(macho_coord.ra), np.median(macho_coord.dec))
distance = center.separation(macho_coord).max()
sep = center.separation(gaia_coord)
temp_gaia = gaia_coord[sep<distance]


corrected = []
factors = []

for p in np.unique(macho[:, [2, 3]], axis=0, return_counts=False):
	c_macho_bool = (macho[:, 2] == p[0]) & (macho[:, 3] == p[1])
	print(p[1])
	if int(p[1])==255:
		corrected.append(np.append(macho[c_macho_bool,-1][:,None], macho_rad[c_macho_bool], axis=1))
		factors.append([0.]*6)
		continue
	print(p)
	c_macho = macho_rad[c_macho_bool]
	c_macho_coord = SkyCoord(c_macho[:, 0], c_macho[:, 1], unit=u.rad)
	temp_corrected = None
	temp_factors = None
	offra = (c_macho[:, 0].max() - c_macho[:, 0].min())
	offdec = (c_macho[:, 1].max() - c_macho[:, 1].min())

	c_temp_gaia = temp_gaia[(temp_gaia.ra.rad < c_macho[:, 0].max() + 1 * offra) &
							(temp_gaia.ra.rad > c_macho[:, 0].min() - 1 * offra) &
							(temp_gaia.dec.rad < c_macho[:, 1].max() + 1 * offdec) &
							(temp_gaia.dec.rad > c_macho[:, 1].min() - 1 * offdec)
							]
	print(len(c_temp_gaia))
	print(len(temp_gaia))
	print(len(c_macho))
	if len(c_temp_gaia)>10000:
		c_temp_gaia = c_temp_gaia[np.random.choice(len(c_temp_gaia), replace=False, size=10000)]

	s2 = np.array(to_cartesian(np.array([c_temp_gaia.ra.rad, c_temp_gaia.dec.rad]).T))
	k = KDTree(s2, metric="euclidean", leaf_size=30)


	def minuit(x):
		ra0, dec0, r, a, alpha, theta = x
		temp = transform(c_macho_coord.ra.rad, c_macho_coord.dec.rad, ra0, dec0, r, a, alpha, theta)
		temp = np.array(to_cartesian(temp))
		d3d = k.query(temp)[0].flatten()
		return 1 / np.sum(1 / (d3d * 180 / np.pi * 3600 + 0.1))  # np.sum(d3d[c])/c.sum()


	bounds = [(c_macho[:, 0].min() - 1 * offra, c_macho[:, 0].max() + 1 * offra),
			  (c_macho[:, 1].min() - 1 * offdec, c_macho[:, 1].max() + 1 * offdec),
			  (0.9, 1.1), (0.9, 1.1), (0, 2 * np.pi), (-5 * np.pi / 180., 5 * np.pi / 180.)]
	i = 0
	pop = 10
	imax = 3
	while i < imax:
		print(i)
		res = scipy.optimize.differential_evolution(minuit, bounds=bounds, popsize=pop, recombination=0.9,
													mutation=(0.3, 0.7), strategy="rand1bin",
													disp=True, maxiter=70)
		print(res)
		res = res.x
		out = transform(c_macho_coord.ra.rad, c_macho_coord.dec.rad, *res)

		correct_macho = SkyCoord(out[:, 0], out[:, 1], unit=u.rad)
		i1, i2, d2d, _ = correct_macho.search_around_sky(c_temp_gaia, seplimit=2 * u.arcsec)
		dra, ddec = correct_macho[i2].spherical_offsets_to(c_temp_gaia[i1])

		if (d2d.arcsec < 1.).sum() / (d2d.arcsec < 2).sum() > 0.4:
			temp_corrected = out
			temp_factors = res
			pop = 10
		if (d2d.arcsec < 0.5).sum() / (d2d.arcsec < 2).sum() > 0.5:
			corrected.append(np.append(macho[c_macho_bool, -1][:, None], out, axis=1))
			factors.append(res)
			break
		i += 1
	if not (temp_corrected is None) and i == imax:
		corrected.append(np.append(macho[c_macho_bool, -1][:, None], temp_corrected, axis=1))
		factors.append(temp_factors)
	if i == imax:
		corrected.append(np.append(macho[c_macho_bool, -1][:, None], macho_rad[c_macho_bool], axis=1))
		factors.append([0.] * 6)
		print("Failed")
	print("MACHO loaded")

out_path = sys.argv[2]
np.savetxt(os.path.join(out_path, "macho_"+str(field)+"_corrected.csv"), np.concatenate(corrected), delimiter=" ", fmt='%s')
print(factors)
print("Done")
