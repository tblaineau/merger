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
def transform(ra, dec, ra0, dec0, r, a, alpha, theta, offra, offdec):
	out = rotation_sphere(ra, dec, ra0, dec0, theta)
	ra1, dec1 = out[:, 0], out[:, 1]
	sina = np.sin(alpha)
	cosa = np.cos(alpha)
	a1 = r * (1 + (a - 1)*cosa**2)
	a2 = r * (a - 1) * cosa * sina
	b2 = r * (1 + (a - 1)*sina**2)
	ra_p = ra0 + a1 * (ra1 - ra0) + a2 * (dec1 - dec0) + offra
	dec_p = dec0 + a2 * (ra1 - ra0) + b2 * (dec1 - dec0) + offdec
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

print("Loading MACHO")
macho =pd.read_csv(MACHO, sep=";", usecols=[0, 1, 2, 3, 4, 6, 7], names=["field", "tile", "id", "ra", "dec", "pier", "chunk"])
ra = macho.ra.str.split(":", expand=True).astype(float)
macho["ra"] = 2*np.pi/24.*(ra[0] + (ra[1]+ra[2]/60.)/60.)
dec = macho.dec.str.split(":", expand=True).astype(float)
macho["dec"]= (np.pi/180. * (np.abs(dec[0]) + (dec[1]+dec[2]/60.)/60.))*np.sign(dec[0])
macho.loc[:, "id_M"] = macho.field.astype(str).str.cat([macho.tile.astype(str), macho.id.astype(str)], sep=":")
macho = macho[(macho.ra!=0) & (macho.dec!=0)]
macho_coord = SkyCoord(macho.ra.values, macho.dec.values, unit=u.rad)
print("Done")

print("Loading Gaia")
gaia = pd.read_csv("/pbs/home/b/blaineau/work/new_association/gaia_edr3_lmc_bright.csv")
gaia_coord = SkyCoord(gaia.ra.values, gaia.dec.values, unit=u.deg, frame="icrs")#, equinox="J2016")
print("Done")

center = SkyCoord(np.median(macho_coord.ra), np.median(macho_coord.dec))
distance = center.separation(macho_coord).max()
sep = center.separation(gaia_coord)
temp_gaia = gaia_coord[sep<distance]


corrected = []
factors = []
pc = np.array(macho.drop_duplicates(["pier", "chunk"])[["pier", "chunk"]])
corrected = []
factors = []

for p in pc:
	c_macho = macho[(macho.pier == p[0]) & (macho.chunk == p[1])]
	c_temp_macho = c_macho.iloc[np.random.randint(0, len(c_macho), 1500)]
	print(p)
	if int(p[1]) == 255:
		corrected.append(np.append(c_macho.id_M[:, None], c_macho[["ra", "dec"]].values, axis=1))
		factors.append([0.] * 6)
		continue
	c_macho_coord = SkyCoord(c_macho.ra.values, c_macho.dec.values, unit=u.rad)
	c_temp_macho_coord = SkyCoord(c_temp_macho.ra.values, c_temp_macho.dec.values, unit=u.rad)
	temp_corrected = None
	temp_factors = None
	offra = (c_macho.ra.max() - c_macho.ra.min())
	offdec = (c_macho.dec.max() - c_macho.dec.min())

	c_temp_gaia = temp_gaia[(temp_gaia.ra.rad < c_macho.ra.max() + 1 * offra) &
							(temp_gaia.ra.rad > c_macho.ra.min() - 1 * offra) &
							(temp_gaia.dec.rad < c_macho.dec.max() + 1 * offdec) &
							(temp_gaia.dec.rad > c_macho.dec.min() - 1 * offdec)
							]
	print(len(c_temp_gaia))
	print(len(temp_gaia))
	print(len(c_macho))
	if len(c_temp_gaia) > 10000:
		c_temp_gaia = c_temp_gaia[np.random.choice(len(c_temp_gaia), replace=False, size=10000)]

	s2 = np.array(to_cartesian(np.array([c_temp_gaia.ra.rad, c_temp_gaia.dec.rad]).T))
	k = KDTree(s2, metric="euclidean", leaf_size=30)


	def minuit(x):
		temp = transform(c_temp_macho_coord.ra.rad, c_temp_macho_coord.dec.rad, *x)
		temp = np.array(to_cartesian(temp))
		d3d = k.query(temp)[0].flatten()
		v = 1 / np.sum(1 / (d3d * 180 / np.pi * 3600 + 0.1))
		print(v, end="\r")
		return v  # np.sum(d3d[c])/c.sum()


	bounds = [(c_macho.ra.min() - 2 * offra, c_macho.ra.max() + 2 * offra),
			  (c_macho.dec.min() - 2 * offdec, c_macho.dec.max() + 2 * offdec),
			  (0.98, 1.02), (0.98, 1.02), (0, 2 * np.pi), (-5 * np.pi / 180., 5 * np.pi / 180.),
			  (-2 * u.arcsec.to(u.rad), 2 * u.arcsec.to(u.rad)), (-2 * u.arcsec.to(u.rad), 2 * u.arcsec.to(u.rad))
			  ]
	i = 0
	pop = 40
	imax = 3
	prec = 0
	while i < imax:
		print(i)
		res = scipy.optimize.differential_evolution(minuit, bounds=bounds, popsize=pop, recombination=0.9,
													mutation=(0.1, 0.3), strategy="randtobest1bin",
													disp=True, maxiter=40)
		print(res)
		res = res.x
		out = transform(c_macho_coord.ra.rad, c_macho_coord.dec.rad, *res)

		correct_macho = SkyCoord(out[:, 0], out[:, 1], unit=u.rad)
		i1, i2, d2d, _ = correct_macho.search_around_sky(c_temp_gaia, seplimit=2 * u.arcsec)
		dra, ddec = correct_macho[i2].spherical_offsets_to(c_temp_gaia[i1])

		tp = (d2d.arcsec < 0.2).sum() / (d2d.arcsec < 2).sum()
		if prec < tp:
			prec = tp
			temp_corrected = out
			temp_factors = res
			print(prec)
			pop = 20
		if (d2d.arcsec < 0.5).sum() / (d2d.arcsec < 2).sum() > 0.85:
			corrected.append(np.append(c_macho.id_M.values[:, None], out, axis=1))
			factors.append(res)
			i = 100
			break
		i += 1
	if i == imax:
		if not (temp_corrected is None):
			corrected.append(np.append(c_macho.id_M.values[:, None], temp_corrected, axis=1))
			factors.append(temp_factors)
		else:
			corrected.append(np.append(c_macho.id_M.values[:, None], c_macho[["ra", "dec"]].values, axis=1))
			factors.append([0.] * 6)
			print("Failed")
	print("MACHO loaded")

out_path = sys.argv[2]
np.savetxt(os.path.join(out_path, "macho_" + str(field) + "_corrected.csv"), np.concatenate(corrected), delimiter=" ",fmt='%s')
print(factors)
print("Done")
