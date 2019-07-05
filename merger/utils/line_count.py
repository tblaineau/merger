#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to compute the total number of lines in MACHO fields
"""


import time
import numpy as np
import os
import argparse

def count_all(filepath):
	st1 = time.time()
	alls=[]
	for filename in os.listdir(filepath):
		print(filename)
		if filename[-3:] == '.gz':
			alls.append((bufcount(filepath, filename), int(filename.split('.')[1])))
	print(time.time()-st1)
	return np.array(alls, dtype=[('linecount', 'i4'), ('tile', 'i4')])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', type=str, required=True, help="Path to MACHO lightcurves files.")
	parser.add_argument('--output-path', type=str, required=False, default='.')

	args = parser.parse_args()

	MACHO_files_path = args.path
	output_path = args.output_path

	for field in range(0, 83):
		print(field)
		np.savetxt(count_all(os.path.join(MACHO_files_path, 'F_'+str(field))))