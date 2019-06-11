import merger.clean.libraries.merger_library as mrgl
import os, time

INPUT_PATH = '/Volumes/DisqueSauvegarde/MACHO/lightcurves'

def test_loading_macho(url=False):
	st1 = time.time()
	t1 = mrgl.read_macho_lightcurve(os.path.join(INPUT_PATH, 'F_1'),'F_1.3319.gz')
	st2 = time.time()
	if url:
		t2 = mrgl.load_macho_from_url('F_1.3319.gz')
		st3 = time.time()

	print(f'Compressed reading time : {st2-st1} seconds for {len(t1)} lines.')
	if url:
		print(f'url reading time : {st3-st2} seconds for {len(t2)} lines.')