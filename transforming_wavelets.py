import pandas as pd
import numpy as np
import os
import  matplotlib.pyplot as plt
from scipy.signal import decimate 
from scipy.signal import medfilt 
from scipy.signal import savgol_filter
import pywt


def load_record(record_name, prefix):
    path = os.path.join(prefix, record_name + ".npy")    
    record = np.load(path)
    return record

def downsample(record):
    # downsample from 500 to 250 hz
    return decimate(record, 2) 

def segmentation(record, hz=250, segment_time=3.072):
    """
        hz -  герц у сигнала
        segment_time - количество секунд в сегменте
        record.shape = [12,2500]
    """
    # так как 2500 не делится на 768 укорачиваю последнюю axis, так чтобы делилось.
    segment_len = int(hz*segment_time)
    chuncks_num = len(record[0])//segment_len
    record = record[:,:int(chuncks_num*segment_len)]

    result = np.asarray(np.split(record,chuncks_num, axis=-1))
    result = np.reshape(result, (12, result.shape[0], result.shape[-1])) # by default shape is 12,3,768
    return result

def pipeline(record_name, prefix):

    record = load_record(record_name, prefix)
    record = downsample(record)
    record = segmentation(record)
    record = medfilt(record, kernel_size=[1,1,3]) #stage 1
    record = medfilt(record, kernel_size=[1,1,3]) #stage 2
    record = savgol_filter(record, window_length = 8, polyorder=3) 

    record = np.asarray(pywt.swt(record, wavelet='db5', level=6, axis=-1))

    return record