#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to compute the total number of lines in MACHO fields
"""


import time
import numpy as np
import os
import argparse
import gzip

def bufcount(filepath, filename):
	st1 = time.time()
	with gzip.open(os.path.join(filepath, filename), 'rt') as f:
		lines = 0
		buf_size = 1024 * 1024
		read_f = f.read # loop optimization

		buf = read_f(buf_size)
		while buf:
			lines += buf.count('\n')
			buf = read_f(buf_size)
		print(time.time()-st1)
		return lines

def count_all(filepath):
	st1 = time.time()
	alls=[]
	for filename in os.listdir(filepath):
		print(filename)
		if filename[-3:] == '.gz':
			alls.append((int(filename.split('.')[1]), bufcount(filepath, filename)))
	print(time.time()-st1)
	return np.array(alls, dtype=[('tile', 'i4'), ('linecount', 'i4')])

def star_count(filepath):
	st1 = time.time()
	alls = []
	for filename in os.listdir(filepath):
		print(filename)
		if filename[-3:] == '.gz':
			alls.append((int(filename.split('.')[1]),
						 len(
							 np.unique(
							 	np.loadtxt(os.path.join(filepath, filename), delimiter=';', usecols=3)
							 ))
						 ))
	print(time.time()-st1)
	return np.array(alls, dtype=[('tile', 'i4'), ('starcount', 'i4')])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', type=str, required=True, help="Path to MACHO lightcurves files.")
	parser.add_argument('--output-path', type=str, required=False, default='.')

	args = parser.parse_args()

	MACHO_files_path = args.path
	output_path = args.output_path

	np.savetxt(os.path.join(output_path,'strcnt_'+str(42)+'.txt'), X=star_count(os.path.join(MACHO_files_path, 'F_' + str(42))), fmt='%d')

	for field in range(1, 83):
		print(field)
		np.savetxt(os.path.join(output_path,'strcnt_' + str(field)+'.txt'), X=star_count(os.path.join(MACHO_files_path, 'F_'+str(field))), fmt='%d')