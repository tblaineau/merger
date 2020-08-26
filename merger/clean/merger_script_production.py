from merger.clean.libraries import merger_library, iminuit_fitter
import argparse, os, logging
import numpy as np
import pandas as pd

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('-t', type=int, required=True)
        parser.add_argument('--correspondance-path', '-pC', type=str, default="/Users/tristanblaineau/")
        parser.add_argument('--output-directory', '-odir', type=str, default=merger_library.OUTPUT_DIR_PATH)
        parser.add_argument('--merged-output-directory', '-modir', type=str, default=".")
        parser.add_argument('--MACHO-field', '-fM', type=int)

        args = parser.parse_args()


        t = args.t
        MACHO_field = args.MACHO_field
        fit = True
        EROS_files_path = 'irods'
        MACHO_files_path = '/sps/hep/eros/data/macho/lightcurves/'
        MACHO_bad_times_directory = "/pbs/home/b/blaineau/work/bad_times/bt_macho"
        correspondance_files_path = args.correspondance_path
        output_directory = args.output_directory
        merged_output_directory = args.merged_output_directory
        verbose = True


        if verbose:
                logging.basicConfig(level=logging.INFO)

        print(fit)

        #Main
        #merged = merger_library.merger_macho_first(merged_output_directory, MACHO_field, EROS_files_path, correspondance_files_path, MACHO_files_path, save=True, t_indice=t)
        print('LOADING MERGED LCs')
        try:
                merged = pd.read_pickle(os.path.join(merged_output_directory, str(MACHO_field)+'_'+str(t)+'.bz2'), compression='bz2')
        except FileNotFoundError:
                print("File not found : "+os.path.join(merged_output_directory, str(MACHO_field)+'_'+str(t)+'.bz2'))
                #print('LOADING FROM EROS-MACHO')
                #merged = merger_library.merger_macho_first(merged_output_directory, MACHO_field, EROS_files_path, correspondance_files_path, MACHO_files_path, save=True, t_indice=t)

        if fit:
                iminuit_fitter.fit_all(merged=merged, filename=str(MACHO_field) + "_" + str(t) + ".pkl", input_dir_path=output_directory, output_dir_path=output_directory)