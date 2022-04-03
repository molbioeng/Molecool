#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal
import math
import statistics
import neurokit2 as nk
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter
import scipy.signal
import peakutils.peak
import seaborn as sns
import padasip as pa
import random


# In[2]:


def powerinterference(signal,rate, amplitude='0.3'):
    """
    The function adds noise characterized by 50 or 60 Hz sinusoidal interference, 
    possibly accompanied by a number of harmonics.
    
    Input: signal-clean signal; rate-sampling rate of the signal.
    
    Output: signal with powerinterference added.
    
    """
    fs = rate
    length = len(signal)
    x = np.arange(length)
    y = amplitude*np.sin(2*np.pi*60 * (x/fs)) 
    signal_interference = signal + y
    
    return signal_interference


# In[3]:


def emgnoise(signal,std='0.05'):
    """
    The function generates EMG noise. It does so by adding random noise from a gaussian distribution with a 
    standard deviation of 0.05. Method used from the literature.
    
    Input: signal-the signal we want to add the noise on.
    
    Ouput: the noisy signal with EMG random noise.
    """
    
    noise = np.random.normal(5,std,len(signal))
    emg = signal + noise
    
    # 5 is the mean of the normal distribution you are choosing from
    # 0.05 is the standard deviation of the normal distribution
    # len(signal) is the number of elements you get in array noise
    
    return emg


# In[4]:


def gen_bw_noise(ecg, fs, amplitude=2.5): # Try with other amplitudes and frequencies.
    """
    Adds baseline wandering to the input signal.

    Parameters:
      fs: Wandering frequency in Hz
      amplitude: Wandering amplitude
      ecg: Original signal
    Output: signal with band-width added
    """
    w = 2*np.pi*fs
    x=np.linspace(0,len(ecg)-1,len(ecg))
    ecg_bw = ecg + amplitude * np.sin(w*x)
    return ecg_bw


# In[5]:


def gen_white_noise(ecg,stan='0.05'):
    """
    Functions which generates white noise and adds it to the signal.
    
    Inputs: ecg-original signal.
    
    Outputs: ecg with white noise.
    
    """
    mean = 0
    std = stan
    num_samples = len(ecg)
    white_noise = 5.2*np.random.normal(mean, std, size=num_samples)
    ecg_wn = ecg + white_noise

    return ecg_wn


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[9]:


def smoothing_window(signal):
    """
    The function creates a sliding window and it is used to smooth the signal out by an average mean.
    Input: signal-the signal stored in an array.
    Output: smoothened signal. 
    """
    #Define window size
    w=31
    #Define mask and store as an array
    mask=np.ones((1,w))/w
    mask=mask[0,:]

    #Convolve the mask with the raw data
    convolved_data=np.convolve(signal,mask,'same')
    return convolved_data


# In[10]:


def removepowerintereference(signal,rate):
    """
    The function removes the power interference line by applying a low-pass filter at 40Hz (as recommended in the literature)
    and a smoothing window afterwards.
    
    Input: signal-noisy signal; rate-sampling rate
    
    Output: deionised signal
    """
    filtered = filter_signal(signal,40,rate)
    convolved = smoothing_window(filtered)
    
    return convolved


# In[11]:


def filteremg(signal,rate):
    """
    The function removes EMG noise. EMG noise is random high-frequency noise to an average moving window filter has
    been used as a candidate to remove it. A high-pass filter is applied in cascade as well.
    
    Input: signal-the noisy signal; rate-the sampling frequency.
    
    Output: filtered signal.
    """
    h = np.full((8, ), 1/8)
    filtered = sp.signal.convolve(signal,h)
    #filtered = np.convolve(signal, np.ones(100)/100, mode='same') # We apply an average moving window to remove high-frequency noise.
    high = filter_signal(filtered,0.5, 1000, order=3, filtertype='highpass') # We apply a high-pass filter.
    
    return filtered


# In[12]:


def remove_baseline_wander(ecg, sample_rate, cutoff=0.06):
    """
    The functions removes the signal's baseline.
    Input: data-signal stored in an array; sample_rate: sample rate in which the signal was sampled; cutoff-frequency frequency from which the values will be filtered out.
    Output: corrected signal.
    """
    return filter_signal(data = ecg, cutoff = cutoff, sample_rate = sample_rate,
                         filtertype='notch')


# In[13]:


def remove_wn(ecg, sample_rate, cutoff=0.6):
    """
    The function removes the white noise from a signal.
    
    Input: ecg-original signal; sample_rate: the frequency at which the signal has been sampled; cutoff- cut off
    frequency for filtering.
    
    Output: signal without noise.
    
    """
    filtered = filter_signal(ecg,cutoff,sample_rate,filtertype='highpass')
    smooth = smoothing_window(filtered)
    
    return smooth


# In[ ]:




