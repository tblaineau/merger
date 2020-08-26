from merger.clean.libraries import merger_library, iminuit_fitter
from merger.clean.libraries.differential_evolution import fit_ml_de_simple
import argparse, os, logging
import numpy as np
import pandas as pd

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('-t', type=int, required=True)
        parser.add_argument('--correspondance-path', '-pC', type=str, default="/Users/tristanblaineau/")
        parser.add_argument('--output-directory', '-odir', type=str, default=merger_library.OUTPUT_DIR_PATH)
        parser.add_argument('--MACHO-field', '-fM', type=int)
        parser.add_argument('--MACHO-bad-time-threshold', '-btt', type=float, default=0.1)

        args = parser.parse_args()


        t = args.t
        MACHO_field = args.MACHO_field
        fit = True
        EROS_files_path = 'irods'
        MACHO_files_path = '/sps/hep/eros/data/macho/lightcurves/'
        merged_output_directory = "/sps/hep/eros/users/blaineau/prod2/merged"
        MACHO_bad_times_directory = "/pbs/home/b/blaineau/work/bad_times/bt_macho"
        correspondance_files_path = "/pbs/home/b/blaineau/data/correspondances"
        bad_times_path = "/pbs/home/b/blaineau/work/bad_times/bt_macho"
        MACHO_btt= args.MACHO_bad_time_threshold
        output_directory = args.output_directory
        verbose = True

        if verbose:
                logging.basicConfig(level=logging.INFO)

        print(fit)

        #Main
        #merged = merger_library.merger_macho_first(merged_output_directory, MACHO_field, EROS_files_path, correspondance_files_path, MACHO_files_path, save=True, t_indice=t)
        logging.info('LOADING MERGED LCs')
        try:
                merged = pd.read_pickle(os.path.join(merged_output_directory, str(MACHO_field)+'_'+str(t)+'.bz2'), compression='bz2')
        except FileNotFoundError:
                logging.info("File not found : "+os.path.join(merged_output_directory, str(MACHO_field)+'_'+str(t)+'.bz2'))
                logging.info('LOADING FROM EROS-MACHO')
                merged = merger_library.merger_macho_first(merged_output_directory, MACHO_field, EROS_files_path, correspondance_files_path, MACHO_files_path, save=True, t_indice=t)

        if fit:
                #Remove bad times
                dfr = []
                dfb = []
                try:
                        df = pd.DataFrame(np.load(os.path.join(bad_times_path, str(MACHO_field) + "_red_M_ratios.npy")),
                                          columns=["red_amp", "time", "ratio"])
                        df.loc[:, "field"] = MACHO_field
                        dfr.append(df)
                        df = pd.DataFrame(np.load(os.path.join(bad_times_path, str(MACHO_field) + "_blue_M_ratios.npy")),
                                          columns=["blue_amp", "time", "ratio"])
                        df.loc[:, "field"] = MACHO_field
                        dfb.append(df)
                except FileNotFoundError:
                        logging.warning("No ratio file found for field "+str(MACHO_field)+".")
                else:
                        dfr = pd.concat(dfr)
                        dfb = pd.concat(dfb)

                        pms = list(zip(merged["time"].values, merged["red_amp"].values))
                        pdf = list(zip(dfr[dfr.ratio>MACHO_btt]["time"].values, dfr[dfr.ratio>MACHO_btt]["red_amp"].values))
                        result = pd.Series(pms).isin(pdf)
                        merged[result].red_M = np.nan
                        merged[result].rederr_M = np.nan

                        pms = list(zip(merged["time"].values, merged["blue_amp"].values))
                        pdf = list(zip(dfr[dfr.ratio>MACHO_btt]["time"].values, dfr[dfr.ratio>MACHO_btt]["blue_amp"].values))
                        result = pd.Series(pms).isin(pdf)
                        merged[result].blue_M = np.nan
                        merged[result].blueerr_M = np.nan

                        merged = merged.dropna(axis=0, how='all', subset=['blue_E', 'red_E', 'blue_M', 'red_M'])

                iminuit_fitter.fit_all(merged=merged,
                                       filename=str(MACHO_field) + "_" + str(t) + ".pkl",
                                       input_dir_path=output_directory,
                                       output_dir_path=output_directory,
                                       fit_function=fit_ml_de_simple,
                                       do_cut5=False,
                                       hesse=True)