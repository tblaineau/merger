import numpy as np
import numba as nb
import sys
from astropy.coordinates import SkyCoord
import astropy.units as u
sys.path.append("/Users/tristanblaineau/Documents/Work/Python")
from lib_perso import *
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize
from iminuit import Minuit

MACHO = "/Volumes/DisqueSauvegarde/MACHO/star_coordinates/DumpStar_15.txt"


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
def transform(ra, dec, ra0, dec0, s, theta):
	out = rotation_sphere(ra, dec, ra0, dec0, theta)
	ra1, dec1 = out[:, 0], out[:, 1]
	dec_p = dec0 + s * (dec1 - dec0)
	ra_p = ra0 + s * (ra1 - ra0)
	return np.array([ra_p, dec_p]).T


def to_minimize(x, f1, cat):
	ra0, dec0, scale, theta = x
	if isinstance(x, u.Quantity):
		ra0, dec0, scale, theta = x.value
	temp = transform(f1.ra.rad, f1.dec.rad, ra0, dec0, scale, theta)
	temp = SkyCoord(temp[:, 0], temp[:, 1], unit=u.rad)
	# tidx = np.random.choice(np.arange(len(cat)), replace=False, size=50000)
	i1, i2, d2d = temp.match_to_catalog_sky(cat)
	# print(d2d.mean(), ra0, dec0, scale, theta)
	return d2d.mean()


macho = []
with open(MACHO) as f:
	for l in f.readlines():
		l  =l.split(";")
		macho.append((l[3], l[4], l[6], l[7], l[8]))
macho = np.array(macho)

macho_rad = []
for i in macho:
	macho_rad.append([right_ascension_to_radians(i[0]), declination_to_radians(i[1])])
macho_rad = np.array(macho_rad)
macho_coord = SkyCoord(macho_rad[:,0], macho_rad[:,1], unit=u.rad)

print("Loading Gaia")
gaia = pd.read_feather("/Users/tristanblaineau/Documents/Work/Jupyter/quad_merge/lmcgaiafull.feather")
print("Done")
gaia_rad = np.array([gaia.ra_epoch2000.values*np.pi/180, gaia.dec_epoch2000.values*np.pi/180]).T
gaia_rad = np.append(gaia_rad, np.arange(0, len(gaia_rad)).reshape(len(gaia_rad), 1), axis=1)
gaia_coord = SkyCoord(gaia.ra_epoch2000.values, gaia.dec_epoch2000.values, unit=u.deg)

center = SkyCoord(np.median(macho_coord.ra), np.median(macho_coord.dec))
distance = center.separation(macho_coord).max()
sep = center.separation(gaia_coord)
temp_gaia = gaia_coord[sep<distance]
#
i1, i2, d2d, _ = macho_coord.search_around_sky(temp_gaia, seplimit=2*u.arcsec)
dra, ddec = macho_coord[i2].spherical_offsets_to(temp_gaia[i1])
plt.hist2d(dra.arcsec, ddec.arcsec, bins=100)
plt.axis("equal")
plt.show()

corrected = []
for p in np.unique(macho[:,[2, 3]], axis=0, return_counts=False)[6:8]:
	print(p)
	c_macho = macho_rad[(macho[:,2] == p[0]) & (macho[:,3] == p[1])]
	c_macho_coord = SkyCoord(c_macho[:,0], c_macho[:,1], unit=u.rad)
	res = scipy.optimize.minimize(to_minimize, [c_macho[0,0], c_macho[0,1], 1., 0.], args=(c_macho_coord, temp_gaia),
								  method="CG")
	out = transform(c_macho_coord.ra.rad, c_macho_coord.dec.rad, *res.x)

	"""def minuit(ra0, dec0, scale, theta):
		temp = transform(c_macho_coord.ra.rad, c_macho_coord.dec.rad, ra0, dec0, scale, theta)
		temp = SkyCoord(temp[:, 0], temp[:, 1], unit=u.rad)
		# tidx = np.random.choice(np.arange(len(cat)), replace=False, size=50000)
		i1, i2, d2d = temp.match_to_catalog_sky(temp_gaia)
		# print(d2d.mean(), ra0, dec0, scale, theta)
		return d2d.mean()

	m = Minuit(minuit, ra0=c_macho[0,0], dec0=c_macho[0,1], scale=1., theta=0., error_ra0=0.001, error_dec0=0.001,
			   error_scale=0.001, error_theta=0.001)
	m.migrad()
	res = m.np_values()
	print(res)"""

	#out = transform(c_macho_coord.ra.rad, c_macho_coord.dec.rad, *res)

	corrected.append(out)
	i1, i2, d2d, _ = c_macho_coord.search_around_sky(temp_gaia, seplimit=2 * u.arcsec)
	dra, ddec = c_macho_coord[i2].spherical_offsets_to(temp_gaia[i1])
	plt.hist2d(dra.arcsec, ddec.arcsec, bins=100)
	plt.axis("equal")
	plt.figure()

	correct_macho = SkyCoord(out[:,0], out[:,1], unit=u.rad)
	i1, i2, d2d, _ = correct_macho.search_around_sky(temp_gaia, seplimit=2 * u.arcsec)
	dra, ddec = correct_macho[i2].spherical_offsets_to(temp_gaia[i1])
	plt.hist2d(dra.arcsec, ddec.arcsec, bins=100)
	plt.axis("equal")
	plt.show()
	print("MACHO loaded")

np.concatenate(corrected)