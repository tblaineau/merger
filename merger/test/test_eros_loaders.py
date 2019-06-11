import os, time

import merger.clean.merger_library as mrgl

#Load test EROS

#Field lm0103n
INPUT_PATH = '/Volumes/DisqueSauvegarde/EROS/lightcurves'

def test_loading_eros(irods=False):
	st1 = time.time()
	t1 = mrgl.load_eros_compressed_files(os.path.join(INPUT_PATH, 'lm/lm010/lm0103n-lc.tar.gz'))
	st2 = time.time()
	t2 = mrgl.load_eros_files(os.path.join(INPUT_PATH, 'lm_ex/lm010/lm0103/lm0103n'))
	st3 = time.time()
	if irods:
		t3 = mrgl.load_irods_eros_lightcurves('/eros/data/eros2/lightcurves/lm/lm010/lm0103/lm0103n')
		st4 = time.time()

	print(f'Compressed reading time : {st2-st1} seconds for {len(t1)} lines.')
	print(f'.time reading time : {st3-st2} seconds for {len(t2)} lines.')
	if irods:
		print(f'iRods reading time : {st4-st3} seconds for {len(t3)} lines.')