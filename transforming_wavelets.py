import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import decimate
from scipy.signal import medfilt
from scipy.signal import savgol_filter
import pywt
import scipy

# copies preprocessing from [https://sci-hub.se/10.1016/j.irbm.2019.09.003]


class BasePreprocessingPipeline():

    def __init__(self, prefix):
        self.prefix = prefix

    def _load_record(self, record_name):
        path = os.path.join(self.prefix, record_name + ".npy")
        record = np.load(path)
        return record

    def _downsample(self, record):
        # downsample from 500 to 250 hz
        return decimate(record, 2)

    def _segmentation(self, record, hz=250, segment_time=3.072):
        """
            hz -  герц у сигнала
            segment_time - количество секунд в сегменте
            record.shape = [12,2500]
        """
        # так как 2500 не делится на 768 укорачиваю последнюю axis, так чтобы делилось.
        segment_len = int(hz*segment_time)
        chuncks_num = len(record[0])//segment_len
        record = record[:, :int(chuncks_num*segment_len)]

        result = np.asarray(np.split(record, chuncks_num, axis=-1))
        # by default shape is 12,3,768
        result = np.reshape(result, (12, result.shape[0], result.shape[-1]))
        return result

    def show_scaleogram(data_sample):
        energy = np.abs(data_sample[0, :, :])**2

        # Create a scaleogram
        plt.figure(figsize=(12, 6))
        plt.imshow(energy, aspect='auto', cmap='jet',
                   extent=[0, energy.shape[0], 1, energy.shape[1]])
        plt.colorbar(label='Energy')
        plt.title('Scaleogram')
        plt.xlabel('Time')
        plt.ylabel('Scale (Level)')
        plt.show()

    def run_preprocessing(self, record_name):
        if (record_name[-2] == "_"):
            record = self._load_record(record_name[:-2])
        else:
            record = self._load_record(record_name)
        record = self._downsample(record)
        record = self._segmentation(record, segment_time=1.5)
        record = medfilt(record, kernel_size=[1, 1, 3])  # stage 1
        record = medfilt(record, kernel_size=[1, 1, 3])  # stage 2
        record = savgol_filter(record, window_length=8, polyorder=3, axis=-1)

        return record


# pipeline for generating features from SWT for catboost
class Pipeline_SWT(BasePreprocessingPipeline):

    def __init__(self, prefix):
        super().__init__(prefix)

    def _calculate_swt_features(self, record):
        record = np.asarray(
            pywt.swt(record, wavelet='db5', level=6, axis=-1))
        features = self._calculate_features(record)

        return features

    def _calculate_features(self, swt_record):
        squared = np.square(swt_record)
        LEE = np.log2(squared).sum()
        SEE = -(squared*np.log2(squared)).sum()
        # NSE нашел там в пдф, не понял что такое subbands, скорее всего просто посчитать все https://www.sciencedirect.com/science/article/pii/S1877050915001313/
        NSE = squared/squared.sum()
        MDS = np.max(np.diff(swt_record))
        MNS = np.diff(swt_record).mean()
        return {"LEE": LEE, "SEE": SEE, "NSE": NSE, "MDS": MDS, "MNS": MNS}

    def run_pipeline(self, record_name):
        record = self.run_preprocessing(record_name)
        record_squared = np.square(record)

        normal_features = self._calculate_swt_features(record)
        squared_features = self._calculate_swt_features(record_squared)

        squared_features = {k+"_squared": v for k,
                            v in squared_features.items()}
        features = normal_features | squared_features
        return features


class Pipeline_CWT_CNN(BasePreprocessingPipeline):
    pass

# def pipeline(record_name, prefix, segment_num, test=False):

#     data_sample = []
#     scales = range(1, 376)
#     # в transforming.ipynb рекорды с миокардом дублируются дважды и к названиям добавляется _1, _2
#     # делается чтобы оверсемплить сэмплы с миокардом, потому что имбаланс.
#     if (record_name[-2] == "_"):
#         for ecg_lead in range(12):
#             seg_num2 = 0 if int(record_name[-1]) == 1 else 2
#             record_part = record[ecg_lead, seg_num2, :]

#             coeffs, freq = pywt.cwt(record_part, scales,
#                                     'morl', sampling_period=1, axis=-1)
#             data_sample.append(coeffs)
#     else:
#         for ecg_lead in range(12):
#             record_part = record[ecg_lead, segment_num, :]

#             coeffs, freq = pywt.cwt(record_part, scales,
#                                     'morl', sampling_period=1, axis=-1)
#             data_sample.append(coeffs)

#     data_sample = np.array(data_sample)

#     if (test):
#         # Calculate the energy (squared magnitude) of the wavelet coefficients
#         # Use one set of coefficients (e.g., approximation coefficients)
#         energy = np.abs(data_sample[0, :, :])**2

#         # Create a scaleogram
#         plt.figure(figsize=(12, 6))
#         plt.imshow(energy, aspect='auto', cmap='jet',
#                    extent=[0, energy.shape[0], 1, energy.shape[1]])
#         plt.colorbar(label='Energy')
#         plt.title('Scaleogram')
#         plt.xlabel('Time')
#         plt.ylabel('Scale (Level)')
#         plt.show()

#     return data_sample


if __name__ == "__main__":
    pipeline('00009_hr', prefix="./train/", segment_num=1)
