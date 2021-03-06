import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gzip, os
import time

def read_macho_lightcurve(filepath):
    try:
        with gzip.open(filepath, 'rt') as f:
            lc = {'time':[], 
                'red_M':[], 
                'rederr_M':[], 
                'blue_M':[], 
                'blueerr_M':[], 
                'id_M':[], 
                # 'red_crowd':[], 
                # 'blue_crowd':[], 
                # 'red_avesky':[],
                # 'blue_avesky':[], 
                # 'airmass':[],
                # 'red_fwhm':[],
                # 'blue_fwhm':[],
                # 'red_normsky':[],
                # 'blue_normsky':[],
                # 'red_chi2':[],
                # 'blue_chi2':[],
                # 'red_type':[],
                # 'blue_type':[]
                }
            for line in f:
                line = line.split(';')
                lc['time'].append(float(line[4]))
                lc['red_M'].append(float(line[9]))
                lc['rederr_M'].append(float(line[10]))
                lc['blue_M'].append(float(line[24]))
                lc['blueerr_M'].append(float(line[25]))
                lc['id_M'].append(line[1]+":"+line[2]+":"+line[3])
                # lc['red_crowd'].append(float(line[13]))
                # lc['blue_crowd'].append(float(line[28]))
                # lc['red_avesky'].append(float(line[20]))
                # lc['blue_avesky'].append(float(line[35]))
                # lc['airmass'].append(float(line[8]))
                # lc['red_fwhm'].append(float(line[21]))
                # lc['blue_fwhm'].append(float(line[36]))
                # lc['red_normsky'].append(float(line[11]))
                # lc['blue_normsky'].append(float(line[26]))
                # lc['red_chi2'].append(float(line[14]))
                # lc['blue_chi2'].append(float(line[29]))
                # lc['red_type'].append(float(line[12]))
                # lc['blue_type'].append(float(line[27]))
            f.close()
    except FileNotFoundError:
        print(filepath+" doesn't exist.")
        return None
    return pd.DataFrame.from_dict(lc)

def load_macho_tiles(MACHO_files_path, field, tile_list):
    macho_path = MACHO_files_path+"F_"+str(field)+"/"
    pds = []
    for tile in tile_list:
        print(macho_path+"F_"+str(field)+"."+str(tile)+".gz")
        # pds.append(pd.read_csv(macho_path+"F_49."+str(tile)+".gz", names=["id1", "id2", "id3", "time", "red_M", "rederr_M", "blue_M", "blueerr_M"], usecols=[1,2,3,4,9,10,24,25], sep=';'))
        pds.append(read_macho_lightcurve(macho_path+"F_"+str(field)+"."+str(tile)+".gz"))
    return pd.concat(pds)

def load_macho_field(MACHO_files_path, field):
    macho_path = MACHO_files_path+"F_"+str(field)+"/"
    pds = []
    for root, subdirs, files in os.walk(macho_path):
        for file in files:
            if file[-2:]=='gz':
                print(file)
                pds.append(read_macho_lightcurve(macho_path+file))
                #pds.append(pd.read_csv(os.path.join(macho_path+file), names=["id1", "id2", "id3", "time", "red_M", "rederr_M", "blue_M", "blueerr_M"], usecols=[1,2,3,4,9,10,24,25], sep=';'))
    return pd.concat(pds)


MACHO_files_path = "/Volumes/DisqueSauvegarde/MACHO/lightcurves/"
field = 24
t = load_macho_field(MACHO_files_path, field)
t.to_pickle('/Volumes/DisqueSauvegarde/working_dir/'+'F_'+str(field)+'.pkl')