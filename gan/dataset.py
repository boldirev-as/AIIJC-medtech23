import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data import Dataset

class DatasetECG(Dataset):
    def __init__(self, annotations_file, signals_dir):
        """
        annotantions_file - path to the annotations dataframe. 
                            First column should be name of the record, second - strat_fold then labels 
        
        signals_dir - path to the directory with transformed signals
        """
        self.signals_labels = pd.read_csv(annotations_file)
        self.signals_dir = signals_dir 

    def __len__(self):
        return len(self.signals_labels)

    def __getitem__(self, idx):
        signals_path = os.path.join(self.signals_dir, self.signals_labels.iloc[idx, 0]+ ".npy")
        signal = np.load(signals_path)
        signal = np.append(signal, [np.zeros(12)]*12, axis=1).astype(np.float32)        


        # iloc[idx, 2:] 2 is because first column is a record name
        labels = torch.from_numpy(self.signals_labels.iloc[idx, 2:].values.astype(int)).float()
        return signal, labels

# class Dataset():
#     def __init__(self, root):
#         self.root = root
#         self.dataset = self.build_dataset()
#         self.length = self.dataset.shape[1]
#         self.minmax_normalize()

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         step = self.dataset[:, idx]
#         step = torch.unsqueeze(step, 0)
#         # target = self.label[idx]
#         target = 0  # only one class
#         return step, target

#     def build_dataset(self):
#         '''get dataset of signal'''
#         dataset = []
#         for _file in os.listdir(self.root):
#             sample = np.loadtxt(os.path.join(self.root, _file)).T
#             dataset.append(sample)
#         dataset = np.vstack(dataset).T
#         dataset = torch.from_numpy(dataset).float()

#         return dataset

#     def minmax_normalize(self):
#         '''return minmax normalize dataset'''
#         for index in range(self.length):
#             self.dataset[:, index] = (self.dataset[:, index] - self.dataset[:, index].min()) / (
#                 self.dataset[:, index].max() - self.dataset[:, index].min())


# if __name__ == '__main__':
#     dataset = Dataset('./data')
#     plt.plot(dataset.dataset[:, 0].T)
#     plt.show()