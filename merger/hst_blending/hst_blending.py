import numba as nb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
import glob
import os

import astropy.units as u
from astropy.coordinates import SkyCoord

#import sys
#sys.path.append('/Users/tristanblaineau/Documents/Work/Python/')
#from lib_perso import proj_ad, print_eros_field_contours, print_macho_fields

from matplotlib.widgets import PolygonSelector
from shapely.geometry import Polygon


with open("/Users/tristanblaineau/Documents/Work/Jupyter/blend/HST/default.param", "r") as f:
    l = f.readlines()

columns = []
for i in l:
    if i[0]!="#":
        columns.append(i.split(" ")[0])
columns[1] = "mag"
columns[2] = "err"
columns[5] = "x"
columns[6] = "y"
columns[13] = "ra"
columns[14] = "dec"


class SelectZone(object):
    def __init__(self, ax):
        self.canvas = ax.figure.canvas
        self.poly = PolygonSelector(ax, self.onselect)
        self.verts = []
    def onselect(self, verts):
        self.verts = verts
    def disconnect(self):
        self.poly.disconnect_events()
        return self.verts


def select_polygon(field):
    fig, ax = plt.subplots()
    plt.scatter(field.ra, field.dec)
    s = SelectZone(ax)
    plt.show()
    g = s.disconnect()
    return g

def shoelace(coo):
    x, y = coo
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


# %matplotlib qt
def load_hst_field(eros_field, hst_field, g=None, timer=15):
	path = "/Users/tristanblaineau/Documents/Work/Jupyter/blend/HST/mast_hst_lmc" + str(eros_field).zfill(3) + "/"
	if len(hst_field) == 8:
		field11 = pd.DataFrame(np.loadtxt(path + "hst_" + hst_field + "_wfpc2_f555w_wf_drz.cat"), columns=columns)
		field12 = pd.DataFrame(np.loadtxt(path + "hst_" + hst_field + "_wfpc2_f814w_wf_drz.cat"), columns=columns)
	else:
		field11 = pd.DataFrame(np.loadtxt(path + hst_field.replace("f814w", "f555w")), columns=columns)
		field12 = pd.DataFrame(np.loadtxt(path + hst_field.replace("f555w", "f814w")), columns=columns)
	if g is None:
		fig, ax = plt.subplots()
		plt.scatter(field11.ra, field11.dec, s=1)
		s = SelectZone(ax)
		plt.show()
		plt.pause(timer)
		g = s.disconnect()
	print(g)
	isinpoly1 = Path(g).contains_points(field11[["ra", "dec"]].values)
	isinpoly2 = Path(g).contains_points(field12[["ra", "dec"]].values)

	field11 = field11[isinpoly1]
	field12 = field12[isinpoly2]

	sk11 = SkyCoord(field11.ra, field11.dec, unit=u.deg)
	sk12 = SkyCoord(field12.ra, field12.dec, unit=u.deg)

	idx1, seps1, _ = sk11.match_to_catalog_sky(sk12)
	idx2, seps2, _ = sk12.match_to_catalog_sky(sk11)

	# Keep only star with closest neighbour counterpart

	c1 = ((idx2[idx1] == np.arange(0, len(sk11))) & (seps1.arcsec < 0.1))
	plt.hist(seps1.arcsec, bins=100, range=(0, 0.2))
	plt.show()
	oks = sk11[c1]

	# offset between two images

	dra, ddec = sk11[c1].spherical_offsets_to(sk12[idx1][c1])

	plt.hist2d(dra.arcsec, ddec.arcsec, bins=100)
	plt.scatter(0, 0, marker='+', s=100, c="r")
	plt.axis("equal")
	plt.show()

	study = pd.concat([field11[c1], field12.iloc[idx1][c1]], axis=1)
	study = field11[c1].reset_index(drop=True).join(field12.iloc[idx1][c1].reset_index(drop=True), rsuffix="_2")
	return study, g


def write_region_file(filename, ra, dec, radius=1.5, color='blue'):
	f = 'global color=' + color + ' dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\n'
	for idx in range(len(ra)):
		f += "circle " + str(ra[idx]) + " " + str(dec[idx]) + " " + str(radius) + '"\n'
	# f +="point("+str(ra[idx])+","+str(dec[idx])+") # point=circle\n"
	file = open(filename, "w")
	file.write(f)
	file.close()


def shoelace(coo):
	x, y = coo
	return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def project(x, ra0):
	return np.array([(x[:, 0] - ra0) * np.cos(x[:, 1] * np.pi / 180.), x[:, 1]]).T


def deproject(x, ra0):
	return np.array([x[:, 0] / np.cos(x[:, 1] * np.pi / 180.) + ra0, x[:, 1]]).T


def inner_limits(g, buffer):
	# g : contours
	# buffer : zone a ôter en degrés
	gc = np.array(g)
	ra0 = gc[:, 0].mean()
	ng = project(gc, ra0)
	poly = Polygon(ng).buffer(-buffer)
	if type(poly) == Polygon:
		ninner = deproject(np.array(poly.exterior.coords), ra0)
	else:
		print("multi")
		ninner = []
		for j in poly:
			ninner.append(deproject(np.array(j.exterior.coords), ra0))
	return ninner


g1 = np.array([(75.33215573485302, -66.01114715397237), (75.36732418245187, -66.02571669639404), (75.40346953137292, -66.0110057021042), (75.43700981010146, -66.02416072584415), (75.36960361887031, -66.05245109947846), (75.29894108989855, -66.0244436295805)])
s1, g1 = load_hst_field("042", "07307_01", g=g1) #EXPTIME = 1200
g2 = np.array([(75.551428962673, -66.08712622910949), (75.65017812623051, -66.08712622910949), (75.64994632068225, -66.12660631065967), (75.6026579888378, -66.12713982527521), (75.60312159993433, -66.10646613392291), (75.55212437931777, -66.10619937661514)])
s2, g2 = load_hst_field("042", "07307_02", g=g2) #EXPTIME = 435

buffer=3.5/3600
inner_s1 = s1[Path(inner_limits(g1, buffer)).contains_points(s1[["ra", "dec"]].values)]
inner_s2 = s2[Path(inner_limits(g2, buffer)).contains_points(s2[["ra", "dec"]].values)]

study = pd.concat([s1, s2])#, s3, s4])
inner_study = pd.concat([inner_s1, inner_s2])#, s3, s4])