from setuptools import setup, find_packages

setup(name='merger',
	  version='0.0a',
	  packages=find_packages(),
	  install_requires=[
		  'numpy',
		  'pandas',
		  'iminuit',
		  'scipy',
		  'matplotlib',
		  'astropy',
		  'numba',
		  'sklearn'
	  ]
	  )