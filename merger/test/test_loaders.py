import os

import merger.old.merger_library as mrgl

#Load test EROS

#Field lm0103n
INPUT_PATH = '/Volumes/DisqueSauvegarde/EROS/lightcurves'

st1 = time.time()
mrgl.load_eros_compressed_files(os.path.join(INPUT_PATH, 'lm/lm010/lm0103n-lc.tar.gz'))
st2 = time.time()
mrgl.load_eros_files(os.path.join(INPUT_PATH, 'lm_ex/lm010/lm0103/lm0103n'))
st3 = time.time()

print(r'Compressed reading time : {st2-st1} s')
print(r'.time reading time : {st3-st2} s')