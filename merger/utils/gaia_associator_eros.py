import numpy as np
import numba as nb
import sys
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import scipy.optimize
from sklearn.neighbors import KDTree
import glob
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
path_to_gaia = "/pbs/home/b/blaineau/work/new_association/gaia_edr3_lmc_bright.csv"
out_path = "/pbs/home/b/blaineau/work/new_association/out/"
gaia = pd.read_csv(path_to_gaia)
gaia_coord = SkyCoord(gaia.ra.values, gaia.dec.values, unit=u.deg, frame="icrs")#, equinox="J2016")

cat = str(field).zfill(3)
for ccd in np.arange(0, 8):
    print(ccd)
    correct_eros=[]
    factors=[]
    vals = []
    for quart in "klmn":
        print(quart)
        stars = []
        path_to_cat = os.path.join( "/sps/eros/data/eros2/lightcurves/lm/lm"+cat+"/lm"+cat+str(ccd)+"/lm"+cat+str(ccd)+quart+"/*.cat")
        pcat = glob.glob(path_to_cat)
        try:
            stars.append(np.genfromtxt(pcat[0]))
        except IndexError:
            print("File doesn't exists.")
        else:
            stars = np.concatenate(stars)
            stars = stars[:, 1:]
            id_E = np.loadtxt(pcat[0], usecols=0, dtype=str)
            print(len(stars))
            eros = pd.DataFrame(stars, columns=["ra", "dec", "mag_R", "magerr_R", "xr", "yr", "mag_B", "magerr_B", "xb", "yb", "varflag"])
            temp_eros = eros.sort_values("mag_B").iloc[:500]
            eros_coord = SkyCoord(eros.ra.values, eros.dec.values, unit=u.deg)
            temp_eros_coord = SkyCoord(temp_eros.ra.values, temp_eros.dec.values, unit=u.deg)

            center = SkyCoord(np.median(eros_coord.ra), np.median(eros_coord.dec))
            distance = center.separation(eros_coord).max()
            sep = center.separation(gaia_coord)
            c_gaia_coord = gaia_coord[(sep<distance)]
            #idx = np.argsort(gaia.phot_g_mean_mag.values)
            temp_gaia = gaia_coord[(sep<distance) & (gaia.phot_g_mean_mag<18)]
            c_temp_gaia = temp_gaia

            def minuit(x):
                temp = transform(temp_eros_coord.ra.rad, temp_eros_coord.dec.rad, *x)
                temp = np.array(to_cartesian(temp))
                d3d = k.query(temp)[0].flatten()
                return 1 / np.sum(1 / (d3d * 180 / np.pi * 3600 + 0.1)) # np.sum(d3d[c])/c.sum()
            offra = (eros_coord.ra.rad.max() - eros_coord.ra.rad.min())
            offdec = (eros_coord.dec.rad.max() - eros_coord.dec.rad.min())

            bounds = [(eros_coord.ra.rad.min() - 2 * offra, eros_coord.ra.rad.max() + 2 * offra),
                      (eros_coord.dec.rad.min()- 2 * offdec ,eros_coord.dec.rad.max()+2 * offdec),
                      (0.98, 1.02), (0.98, 1.02), (0, 2 * np.pi), (-5 *np.pi/180., 5 *np.pi/180.),
                      (-2*u.arcsec.to(u.rad), 2*u.arcsec.to(u.rad)),  (-2*u.arcsec.to(u.rad), 2*u.arcsec.to(u.rad))
                     ]

            if len(c_temp_gaia)>10000:
                c_temp_gaia = c_temp_gaia[np.random.choice(len(c_temp_gaia), replace=False, size=10000)]

            s2 = np.array(to_cartesian(np.array([c_temp_gaia.ra.rad, c_temp_gaia.dec.rad]).T))
            k = KDTree(s2, metric="euclidean", leaf_size=30)

            i1, i2, d2d, _ = temp_eros_coord.search_around_sky(c_gaia_coord, seplimit=2 * u.arcsec)
            dra, ddec = temp_eros_coord[i2].spherical_offsets_to(c_gaia_coord[i1])

            i = 0
            pop = 70
            imax = 6
            prec = 0
            while i < imax:
                print(i)
                def minuit(x):
                    temp = transform(temp_eros_coord.ra.rad, temp_eros_coord.dec.rad, *x)
                    temp = np.array(to_cartesian(temp))
                    d3d = k.query(temp)[0].flatten()
                    return 1 / np.sum(1 / (d3d * 180 / np.pi * 3600 + 0.1))

                res = scipy.optimize.differential_evolution(minuit, bounds=bounds, popsize=pop, recombination=0.7,
                                                                mutation=(0.1, 0.3), strategy="randtobest1bin",
                                                                disp=True, maxiter=40)
                res = res.x
                out = transform(eros_coord.ra.rad, eros_coord.dec.rad, *res)

                c_eros = SkyCoord(out[:, 0], out[:, 1], unit=u.rad)
                i1, i2, d2d, _ = c_eros.search_around_sky(c_gaia_coord, seplimit=2 * u.arcsec)
                dra, ddec = c_eros[i2].spherical_offsets_to(c_gaia_coord[i1])


                if (d2d.arcsec < 1.).sum() / (d2d.arcsec < 2).sum() > 0.4:
                    tp = (d2d.arcsec < 0.2).sum() / (d2d.arcsec < 2).sum()
                    if not prec or prec < tp:
                    #    t = transform(temp_eros_coord.ra.rad, temp_eros_coord.dec.rad, *res)
                    #    temp_eros_coord = SkyCoord(t[:, 0], t[:, 1], unit=u.rad)
                        prec = tp
                        temp_corrected = out
                        temp_factors = res
                        print(prec)
                    pop = 20
                if (d2d.arcsec < 0.25).sum() / (d2d.arcsec < 2).sum() > 0.75:
                    correct_eros.append(np.append(id_E[:,None], out, axis=1))
                    factors.append(res)
                    vals.append(0)
                    i=100
                    break
                i+=1
            if i == imax:
                if not (temp_corrected is None):
                    correct_eros.append(np.append(id_E[:,None], temp_corrected, axis=1))
                    factors.append(temp_factors)
                    vals.append(1)
                else:
                    correct_eros.append(np.append(id_E[:,None], eros_coord, axis=1))
                    factors.append([0.] * 6)
                    vals.append(2)
                    print("Failed")
    np.savetxt(os.path.join(out_path, "validation_lm"+cat+str(ccd)+".csv"), vals)
    np.savetxt(os.path.join(out_path, "eros_lm"+cat+str(ccd)+"_corrected.csv"), np.concatenate(correct_eros), delimiter=" ", fmt='%s')
