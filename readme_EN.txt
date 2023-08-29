How to open signal files:
import numpy as np
with open("path/train/*file name*.npy", "rb") as f:
    data  = np.load(f, allow_pickle=True)

How to open csv files:
import pandas as pd
data = pd.read_csv("path/*file_name*.csv")

Additional information about the files:
- *_meta.csv – metadata for each record taken from the official repository https://physionet.org/content/ptb-xl/1.0.3/ (supplemented with numbers of groups of doctors that marked-up the data)
