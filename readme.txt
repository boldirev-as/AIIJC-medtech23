Как открывать файлы с сигналами:

import numpy as np
with open("path/train/*имя файла*.npy", "rb") as f:
    data  = np.load(f, allow_pickle=True)

Как открывать csv-файлы:

import pandas as pd
data = pd.read_csv("path/*имя_файла*.csv")


Дополнительная информация о файлах:
 
- *_meta.csv - метаданные про каждую запись, взятые из официального репозитория https://physionet.org/content/ptb-xl/1.0.3/ (к ним добавлены номера групп врачей, которые размечали данные)
