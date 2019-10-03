import numpy as np
import os
import ssl
from irods.session import iRODSSession
from irods.exception import CollectionDoesNotExist, DataObjectDoesNotExist
import logging
import subprocess
import sys


root_irods_filepath = "/eros/data/eros2/fits/lm/"

def get_fits_from_irods(id_E, color, index_list, save_directory):
	field = id_E[:5]
	ccd = id_E[5]
	if color=='r':
		color_i = "0"
	else:
		color_i = "1"
	irods_filepath = os.path.join(root_irods_filepath, field, field+color_i+str(ccd))
	try:
		env_file = os.environ['IRODS_ENVIRONMENT_FILE']
	except KeyError:
		env_file = os.path.expanduser('~/.irods/irods_environment.json')

	ssl_context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH, cafile=None, capath=None, cadata=None)
	ssl_settings = {'ssl_context': ssl_context}
	pds = []
	with iRODSSession(irods_env_file=env_file, **ssl_settings) as session:
		try:
			coll = session.collections.get(irods_filepath)
		except CollectionDoesNotExist:
			logging.error(f"iRods path not found : {irods_filepath}")

		for index in index_list:
			filename = field+str(color_i)+str(ccd)+'t'+color+'r'+index+'.fits'
			#TODO: Check for existence
			full_irods_filepath = os.path.join(irods_filepath, filename)
			try:
				obj = session.data_objects.get(full_irods_filepath)
			except DataObjectDoesNotExist:
				logging.error(f"iRods file not found : {full_irods_filepath}")

			logging.info(f"OPENING FITS ({filename})")
			with obj.open('r') as f:
				logging.info('WRITING LOCALLY')
				#open(os.path.join(save_directory, filename), "wb").write(f.read())
				with open(os.path.join(save_directory, filename), "wb") as f_output:
					while True:
						chunk = f.read(1048576)
						f_output.write(chunk)
						if not chunk:
							break


	return 0


def load_MACHO_fits():
	oid = int(t1[t1.time==curr_time].observation_id.iloc[0])
	url = 'http://macho.nci.org.au/macho_images/O_'
	file = 'Obs_'+str(oid)+'-WCS-MEF.tar.gz'
	print(file)
	urlpath = url+str(oid//1000)+'/Obs_'+str(oid)+'-WCS-MEF.tar.gz'
	if not os.path.isfile(file):
		file = wget.download(urlpath)
	f = tarfile.open(file, 'r:gz')
	blue_f, red_f = f.getmembers()
	hdul = fits.open(f.extractfile(blue_f))


def astrometrize(original_fits_directory, astrometry_fits_directory, field_center, fits_file):
	cmd = ' '.join(["/usr/local/astrometry/bin/solve-field",
					os.path.join(original_fits_directory, fits_file),
					"--fits-image",
					"--ra " + str(field_center[0]),
					"--dec " + str(field_center[1]),
					"--radius 1",
					" --overwrite",
					"-L 19 -H 22 -u 'aw' -z 2",
					"-D " + astrometry_fits_directory,
					"-d 100",
					"-N "+os.path.join(astrometry_fits_directory, fits_file),
					"--no-plots",
					"--skip-solved",
					"-C "+os.path.join(astrometry_fits_directory, fits_file)]
				   + ["-"+letter+" 'none'" for letter in "MRBWP"])
	print(cmd)
	try:
		print(subprocess.check_output(cmd, shell=True))
	except subprocess.CalledProcessError:
		logging.error("Process crashed, continuing")
		return 1

	with fits.open(os.path.join(astrometry_fits_directory, fits_file)) as hdul:
		hdu = hdul[0]
		position = astropy.coordinates.SkyCoord(1.42584761, -1.20748266, unit=u.rad, frame='icrs')
		wcs = astropy.wcs.WCS(hdu.header)
		size = (60, 60)
		cutout = Cutout2D(hdu.data, position, size=size, wcs=wcs)
		hdu.data = cutout.data
		hdu.header.update(cutout.wcs.to_header())
		cutout_filename = 'cutted_'+fits_file
		hdu.writeto(os.path.join(astrometry_fits_directory, cutout_filename), overwrite=True)

	cmd = ' '.join(["/usr/local/astrometry/bin/solve-field",
					os.path.join(original_fits_directory, fits_file),
					"--fits-image",
					" --overwrite",
					"-D " + astrometry_fits_directory,
					"-d 100",
					"-N "+os.path.join(astrometry_fits_directory, 'astr_'+cutout_filename),
					"--no-plots",
					"--skip-solved",
					"-C "+os.path.join(astrometry_fits_directory, 'astr_'+cutout_filename)]
				   + ["-"+letter+" 'none'" for letter in "MRBWP"])
	print(cmd)
	try:
		print(subprocess.check_output(cmd, shell=True))
	except subprocess.CalledProcessError:
		logging.error("Process crashed on cutout run, continuing")
		return 2

logging.basicConfig(level=logging.INFO)

index_list = ['8k2165', '9c1540', '9c1931', '9c2632', '9d0438', '9i2073',
       '9i2977', '9j0785', '9l1433', '9l1831', '9l2648', 'aa0536',
       'aa0934', 'aa1435', 'aa1950', 'aa2338', 'aa2746', 'aa3130',
       'ab0449', 'ab0854', 'ab1248', 'ab1645', 'ab2018', 'ab2519',
       'ac0336', 'ac0934', 'ac1529', 'ac2126', 'ac2829', 'ad0719',
       'ad2035', 'ad2632', 'ag05211']

original_fits_directory = '/Users/tristanblaineau/Documents/Work/Python/film_maker/lm0011m23360'
astrometry_fits_directory = os.path.join(original_fits_directory, 'done')
#get_fits_from_irods('lm0306n19520', 'b', index_list, original_fits_directory)

fields_coordinates = np.loadtxt('/Volumes/DisqueSauvegarde/EROS/lm/lm.field', usecols=[1, 2, 3, 4], skiprows=1)
fields_centers = np.array([(fields_coordinates[:,0]+fields_coordinates[:,1])/2, (fields_coordinates[:,2]+fields_coordinates[:,3])/2]).T
fields_centers[0,0] = 81.22
fields_centers[0,1] = -69.23
print(fields_centers[0])

to_read = ['8k2165', '9c1540', '9c1931', '9c2632', '9d0438', '9i2073',
       '9i2977', '9j0785', '9l1433', '9l1831', '9l2648', 'aa0536',
       'aa0934', 'aa1435', 'aa1950', 'aa2338', 'aa2746', 'aa3130',
       'ab0449', 'ab0854', 'ab1248', 'ab1645', 'ab2018', 'ab2519',
       'ac0336', 'ac0934', 'ac1529', 'ac2126', 'ac2829', 'ad0719',
       'ad2035', 'ad2632', 'ag05211']
for idx, filename in enumerate(to_read):
	to_read[idx] = 'lm03016tbr'+filename+'.fits'

"""for fits_file in to_read:# = os.listdir('/Users/tristanblaineau/Documents/Work/Python/film_maker/lm0306n19520')[1]
	cmd = ' '.join(["/usr/local/astrometry/bin/solve-field",
					os.path.join(original_fits_directory, fits_file),
					"--fits-image",
					"--ra 75.9104",#+str(fields_centers[29,0]),
					"--dec -67.4849",#+str(fields_centers[29,1]),
					" --overwrite",
					"--radius 1 -L 19 -H 22 -u 'aw' -z 2",
					"-D " + astrometry_fits_directory,
					"-d 100",
					"-N "+os.path.join(astrometry_fits_directory, fits_file),
					"--no-plots",
					"--skip-solved",
					"-C "+os.path.join(astrometry_fits_directory, fits_file)] + ["-"+letter+" 'none'" for letter in "MRBWP"])
	print(cmd)
	try:
		print(subprocess.check_output(cmd, shell=True))
	except subprocess.CalledProcessError:
		logging.error("Process crashed, continuing")"""



from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import astropy
import astropy.visualization
from astropy.nddata import Cutout2D
import astropy.units as u
import matplotlib.animation as animation
import scipy.optimize
from iminuit import Minuit

fig=plt.figure()
padded_reference = 'a'
ims = list()
zscale = astropy.visualization.ZScaleInterval()
for fits_file in os.listdir(astrometry_fits_directory):
	if fits_file[-5:]==".fits":
		print(fits_file)
		with fits.open(os.path.join(astrometry_fits_directory, fits_file)) as hdul:
			hdu = hdul[0]
			position = astropy.coordinates.SkyCoord(1.42584761, -1.20748266, unit=u.rad, frame='icrs')
			wcs = astropy.wcs.WCS(hdu.header)
			size = (60, 60)
			pad_width = 7
			cutout = Cutout2D(hdu.data, position, size=size, wcs=wcs)
			vmin, vmax = zscale.get_limits(cutout.data)
			current_norm = colors.PowerNorm(2, vmin=vmin, vmax=vmax)
			if padded_reference == 'a':
				padded_reference = np.pad(cutout.data, pad_width, "constant", constant_values=0)
				shifted = padded_reference
			else:
				def abs_difference(x):
					x_shift, y_shift = x
					x_shift = int(np.round(x_shift))
					y_shift = int(np.round(y_shift))
					if x_shift>2*pad_width or y_shift>2*pad_width:
						return np.inf
					shifted = np.zeros((size[0] + pad_width*2, size[1] + pad_width*2))
					shifted[x_shift:x_shift+size[0], y_shift:y_shift+size[1]] = current_norm(cutout.data)
					xlow = max(x_shift, pad_width)
					xmax = min(x_shift+size[0], size[0]+pad_width)
					ylow = max(y_shift, pad_width)
					ymax = min(y_shift+size[1], size[1]+pad_width)
					intersection_size = np.abs((xlow-xmax)*(ylow-ymax))
					#intersection_size = size[0]*size[1] - np.abs((x_shift-size[0])*(y_shift-size[1]))
					#plt.imshow((1/np.abs((shifted-padded_reference)))[xlow:xmax, ylow:ymax])
					#plt.show()
					return -(np.abs(1/padded_reference*(shifted-padded_reference)))[xlow:xmax, ylow:ymax].sum()/intersection_size
				"""m = Minuit(abs_difference,
						   x_shift=pad_width,
						   y_shift=pad_width,
						   limit_x_shift=(0, 2*pad_width),
						   limit_y_shift=(0, 2*pad_width),
						   error_x_shift=5,
						   error_y_shift=5,
						   errordef = 1,
						   print_level=0
				)
				m.migrad()
				x_shift, y_shift = (np.round(m.values.values())).astype(int)"""
				res, x0, grid, Jout = scipy.optimize.brute(abs_difference, ranges=((1, 2*pad_width-1), (1, 2*pad_width-1)), full_output=True)
				x, y = grid
				print(res, x0)
				#plt.pcolormesh(x, y, Jout)
				#plt.show()
				x_shift, y_shift = (np.round(res)).astype(int)

				shifted = np.zeros((size[0] + pad_width * 2, size[1] + pad_width * 2))
				shifted[x_shift:x_shift + size[0], y_shift:y_shift + size[1]] = cutout.data

			vmin,vmax = zscale.get_limits(cutout.data)
			im = plt.imshow(shifted[pad_width*2:size[0]+pad_width, pad_width*2:size[1]+pad_width], origin='lower', norm=colors.PowerNorm(2, vmin=vmin, vmax=vmax), cmap='Greys_r')
			ims.append([im])
			plt.show()

anim = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=5000)
#anim.save("animation_test.gif")
plt.show()