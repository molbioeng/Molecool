#!/usr/bin/env python
# coding: utf-8

# # Functions for detecting peaks from ECG signals
# The file below contains all the functions needed to detect the peaks from an ECG signal acuretely. It starts by filtering and enhacning the signal, detecting the peaks based on a threshold and then correcting the detected peaks.

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math
import statistics
import neurokit2 as nk
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter
import scipy.signal
import peakutils.peak
import seaborn as sns
import pywt as pw


# In[2]:


def butter_lowpass(cutoff, sample_rate, order=2):
    """
    The function returns the butter indexes for a butter lowpass filter. Github https://github.com/paulvangentcom/heartrate_analysis_python/tree/0005e98618d8fc3378c03ab0a434b5d9012b1221 
    
    Input: cutoff-frequency from which the values will be filtered out; order-stregnth of the filter; sample_rate-rate at which the signal was sampled.
    
    Ouput: butter indeces.

    """
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass(cutoff, sample_rate, order=2):
    """
    The function returns the butter indexes for a highpass filter. Github https://github.com/paulvangentcom/heartrate_analysis_python/tree/0005e98618d8fc3378c03ab0a434b5d9012b1221 
    
    Input: cutoff-frequency from which the values will be filtered out; order-stregnth of the filter; sample_rate-rate at which the signal was sampled.
    
    Ouput: butter indeces.

    """
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


# In[3]:


def filter_signal(data, cutoff, sample_rate, order=2, filtertype='lowpass'):
    """
    The function filters data in the frequency domain. 
    
    Input: data-signal data stored in an array; cutoff-frequency from which the values will be filtered out; order-stregnth of the filter; filtertype-type of filter:
        lowpass,highpass,bandpass,notch
    
    Ouput: filtered_data

    """
    
    if filtertype.lower() == 'lowpass':
        b, a = butter_lowpass(cutoff, sample_rate, order=order)
    elif filtertype.lower() == 'highpass':
        b, a = butter_highpass(cutoff, sample_rate, order=order)
    elif filtertype.lower() == 'bandpass':
        assert type(cutoff) == tuple or list or np.array, 'if bandpass filter is specified, cutoff needs to be array or tuple specifying lower and upper bound: [lower, upper].'
        b, a = butter_bandpass(cutoff[0], cutoff[1], sample_rate, order=order)
    elif filtertype.lower() == 'notch':
        b, a = iirnotch(cutoff, Q = 0.005, fs = sample_rate)
    else:
        raise ValueError('filtertype: %s is unknown, available are: lowpass, highpass, bandpass, and notch' %filtertype)

    filtered_data = filtfilt(b, a, data)
    
    return filtered_data


# In[4]:


def remove_baseline_wander(data, sample_rate, cutoff=0.05):
    """
    The functions removes the signal's baseline.
    Input: data-signal stored in an array; sample_rate: sample rate in which the signal was sampled; cutoff-frequency frequency from which the values will be filtered out.
    Output: corrected signal.
    """
    return filter_signal(data = data, cutoff = cutoff, sample_rate = sample_rate,
                         filtertype='notch')


# In[5]:


def cos_correction(signal):
    """
    The function removes a tenth of both ends from the signal by multiplying it by a half cosine function.
    Input: signal-index of data points to be corrected.
    Output: even signal with the corners diminished. 
    """
    
    length = len(signal) # We find the length of the signal.
    signal_10 = 0.6*length # We calculate the length of a tenth of the signal.
    signal_90 = length - signal_10 # We calculate the difference to know the tenth corresponding to the array's final end.
    step = (np.pi/2)/signal_10 # We find the number of values our cosine function needs to have.
    x = np.arange(0,np.pi/2,step) # We generate half a cosine function: start,stop,step.
    y = np.cos(x-1.57) # We shift it by pi so we can use it as a smoothening factor to apply to the values at the end of the signal.
    cos_init = y
    cos_end = y
    cos_end[-1] = 0
    cos_end[-2] = 0
    cos_end[-3] = 0
    
    for index,value in enumerate(cos_init):
        signal[index] = signal[index]*value # We perform the multiplicatoin by the signal points and the cosine points.

    for index, value in enumerate(cos_end):
        signal[-index] = value * signal[-index]
    return signal


# In[6]:


def scale_data(data, lower=0, upper=1024):
    """
    Subfunction from the enhance_peaks section. It allows to scale the data.
    Input: data-the signal, lower,upper-ranges that describe the scaling factor.
    Output: scaled signal. 
    """
    rng = np.max(data) - np.min(data)
    minimum = np.min(data)
    data = (upper - lower) * ((data - minimum) / rng) + lower
    return data


# In[7]:


def enhance_peaks(hrdata, iterations=2):
    """
    The function squares the signal and, hence, enhances the peaks.
    Input: hrdata-the signal stored in an array; iterations-the times the signal is squared.
    Output: enhanced signal. 
    """
    scale_data(hrdata)
    for i in range(iterations):
        hrdata = np.power(hrdata, 2)
        hrdata = scale_data(hrdata)
    return hrdata 


# In[8]:


def smoothing_window(signal):
    """
    The function creates a sliding window and it is used to smooth the signal out by an average mean.
    Input: signal-the signal stored in an array.
    Output: smoothened signal. 
    """
    #Define window size
    w=30
    #Define mask and store as an array
    mask=np.ones((1,w))/w
    mask=mask[0,:]

    #Convolve the mask with the raw data
    convolved_data=np.convolve(np.squeeze(signal),np.squeeze(mask),'same')
    return convolved_data


# In[9]:


def _filtering(signal,rate):
    """
    Application of a sub-sequent filtering steps, based on the Pan-Tomkins peak scalation for subsequent peak detection. 
    Input: signal-the signal stored in an array; rate-the sampling frequency at which the signal was sampled.
    Output: filtered signal. 
    """
    normalised = (signal - np.min(signal)) / np.max(signal-np.min(signal)) # Signal normalisation.
    #cos_removed = cos_correction(signal) # We remove the points located at both ends of the signal.
    low = filter_signal(normalised, 15, rate, order=4, filtertype='lowpass') # We apply a low-pass filter.
    high = filter_signal(low,5, rate, order=4, filtertype='highpass') # We apply a high-pass filter.
    coeffs = pw.swt(high, wavelet = "haar", level=2, start_level=0, axis=-1)
    wv = coeffs[1][1] ##2nd level detail coefficients
    convolved = smoothing_window(wv)
    remove = remove_baseline_wander(convolved, rate) # We remove the signal's baseline.
    en = enhance_peaks(remove, iterations=2) # We enhance the signal twice to improve the following peak-detection.
    smoothing = smoothing_window(en)

    return smoothing


# In[10]:


def _peakdetection(signal):
    
    """
    
    The funtion detects the peaks based on a threshold algorithm described in the following paper:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7922324/. The algorithm has been tunned to incorporate a sliding threshold
    that accounts for enhanced peaks. 
    
    Input: signal-the signal we would like to obtain the peaks from.
    Output: a dataframe containing the x and y values corresponding to the different peaks detected.
    
    """
    
    peaks = pd.DataFrame()
    y_points = [] # List where we are going to store the y_values of the peaks detected.
    x_points = [] # List where we are going to store the x_values of the peaks detected.
    temp = 0.5*((0.75*np.percentile(signal, 90))+(0.25*np.mean(signal))) #+ np.std(signal) # We define the threshold.
    for index,point in enumerate(signal):
        if index+1 == len(signal):
            break
        if point > temp: # We check if the value is bigger than the threshold. If it is the case, we set it as the new threshold.
            threshold = point
            if signal[index+1]< threshold and (signal[index]-signal[index-1]) > 0:# If the next value is lower than the threshold, then we add the point as detected and we restart the threshold.
                if point > signal[index-1] and point > signal[index+1]:
                    y_points.append(point) # We add the difference between the values because the first condition will detect all the points on the QRS complex, we need to not save the lower points on the peak's slope.
                    x_points.append(index)
        if point < temp: # Once the peak is stored, we restart the threshold value to calculate the following peak.
            threshold = temp
            
    diff = np.diff(x_points)
    x_del = []
    y_del = []
    for index,value in enumerate(diff):
        if value < 200:
            x_del = np.append(x_del,index)
    
    x_del = [ int(val) for val in x_del ]
   
    x_points = np.delete(x_points,x_del)
    y_points = np.delete(y_points,x_del)

    peaks['x_values'] = x_points
    peaks['y_values'] = y_points
    
    return peaks

    


# In[11]:


def _peakcorrection(peaks,or_signal,rate):
    """
    Function that correct the detected peaks to be found exactly at the maximum point of the QRS complex by using a slinding window.
    Inputs: peaks - the dictionary containing the detected peaks; or_signal - the original signal inside a 1D array; rate - sampling rate of the signal.
    Output: peask - corrected peaks inside a dictionary
    """
    
    ## Pin point exact qrs peak
    window_check = int(rate/6)
    #signal_normed = np.absolute((signal-np.mean(signal))/(max(signal)-min(signal)))
    r_peaks = [0]*len(peaks['x_values'])
    
    for i,loc in enumerate(peaks['x_values']):
        start = max(0,loc-window_check)
        end = min(len(or_signal),loc+window_check)
        wdw = np.absolute(or_signal[start:end] - np.mean(or_signal[start:end]))
        pk = np.argmax(wdw)
        r_peaks[i] = start+pk

    y_values = []
    for value in r_peaks:
        y_values.append(or_signal[value])
        
    peaks = pd.DataFrame()
    peaks['x_values'] = r_peaks
    peaks['y_values'] = y_values
    
    return peaks
    


# In[12]:


def filteringdet(signal,rate):
    """
    Function wich combines all the filtering and peak detection functions in one module.
    """
    normalised = (signal - np.min(signal)) / np.max(signal-np.min(signal))
    filtered = _filtering(signal,rate)
    peaks = _peakdetection(filtered)
    n_peaks = _peakcorrection(peaks,signal,rate)
    
    return n_peaks


# In[ ]:




