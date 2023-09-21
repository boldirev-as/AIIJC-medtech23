import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pywt
from preprocessing_with_beats.preprocessing_beat import extract_beats


class PipelineBeatExtraction():
    def __init__(self, prefix, frequency=500, num_secs_for_beat=1, noise_level=3):
        """
            prefix - путь к папке, где хранятся данные 
            frequency - частота сигнала 
            num_secs_for_beat - длина удара сердца в секундах
            noise_level - степень удаления шума(до какого уровня wavelet decomposition считать шумом)
        """
        self.prefix = prefix
        self.db6 = pywt.Wavelet("db6")
        self.freq = frequency
        self.num_secs_for_beat = num_secs_for_beat
        self.noise_level = noise_level

    def _load_record(self, record_name):
        path = os.path.join(self.prefix, record_name + ".npy")
        record = np.load(path)
        return record

    def _denoise(self, record):

        # имеет вид [cA_n, c_Dn, ... , C_D1],
        # где cA_n - aproximation coefficients последнего уровня, они нам не нужны, cD_n - detail coefficients n уровня.
        coefs = pywt.wavedec(record, self.db6, axis=-1)

        # убираем aproximation coeficients
        coefs[0] = np.zeros_like(coefs[0])

        # обнуляем начальные левелы, так написано в папире про денойзинг,
        # так как экг сигнал с частотой >45 hz имеет мало информации
        for i in range(1, self.noise_level+1):
            coefs[-i] = np.zeros_like(coefs[-i])
        denoised = pywt.waverec(coefs, self.db6)

        return denoised

    def _extract_beats(self, denoised_record):
        return extract_beats(denoised_record, self.freq, self.num_secs_for_beat)

    def run_pipeline(self, record_name):

        record = self._load_record(record_name)
        denoised_record = self._denoise(record)
        beats = self._extract_beats(denoised_record)

        min_beat_len = len(beats[0])
        for ecg_channel in beats:
            min_beat_len = min(len(ecg_channel), min_beat_len)
        
        for ecg_channel in range(len(beats)):
            beats[ecg_channel] = beats[ecg_channel][:min_beat_len]
        # отбрасываем удары, чтобы все каналы ЭКГ были одинаковой длины. 
        # TODO: решить, каким образом сохранять эти удары, мб переписать annotations.
        return np.asarray(beats)
