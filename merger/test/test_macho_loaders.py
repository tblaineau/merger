import merger.clean.merger_library as mrgl
import time

def test_loading_macho(url=False):
	st1 = time.time()
	mrgl.load
	if url:
		mrgl.load_macho_from_url('F_1.3319.gz')