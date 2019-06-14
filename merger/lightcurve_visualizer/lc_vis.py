import pandas as pd
import numpy as np
import os

WORKING_DIR =  "/Volumes/DisqueSauvegarde/working_dir/"
FILENAME = "5_lm0103.pkl"

pd.from_pickle(os.path.join(WORKING_DIR, FILENAME))
