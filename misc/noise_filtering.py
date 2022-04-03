import numpy as np
import scipy as sp
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter




def butter_lowpass(cutoff, sample_rate, order=2):
    """
    The function returns the butter indexes for a butter lowpass filter. Github https://github.com/paulvangentcom/heartrate_analysis_python/tree/0005e98618d8fc3378c03ab0a434b5d9012b1221 
    
    Input: cutoff-frequency from which the values will be filtered out; order-stregnth of the filter; sample_rate-rate at which the signal was sampled.
    
    Ouput: butter indeces.

    """
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return butter(order, normal_cutoff, btype='low', analog=False)




def butter_highpass(cutoff, sample_rate, order=2):
    """
    The function returns the butter indexes for a highpass filter. Github https://github.com/paulvangentcom/heartrate_analysis_python/tree/0005e98618d8fc3378c03ab0a434b5d9012b1221 
    
    Input: cutoff-frequency from which the values will be filtered out; order-stregnth of the filter; sample_rate-rate at which the signal was sampled.
    
    Ouput: butter indeces.

    """
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return butter(order, normal_cutoff, btype='high', analog=False)





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
        assert type(cutoff) == tuple or list or np.array, 'if bandpass filter is specified, \
cutoff needs to be array or tuple specifying lower and upper bound: [lower, upper].'
        b, a = butter_bandpass(cutoff[0], cutoff[1], sample_rate, order=order)
    elif filtertype.lower() == 'notch':
        b, a = iirnotch(cutoff, Q = 0.005, fs = sample_rate)
    else:
        raise ValueError('filtertype: %s is unknown, available are: \
lowpass, highpass, bandpass, and notch' %filtertype)

    filtered_data = filtfilt(b, a, data)
    
    return filtered_data




def smoothing_window(signal, w=31):
    """
    The function creates a sliding window and it is used to smooth the signal out by an average mean.
    Input: signal-the signal stored in an array.
    Output: smoothened signal. 
    """

    #Define mask and store as an array
    mask = np.ones((1,w))/w
    mask = mask[0,:] #?

    #Convolve the mask with the raw data
    convolved_data = np.convolve(signal,mask,'same')
    return convolved_data



# Weird name - must be a typo
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



def filteremg(signal,rate):
    """
    The function removes EMG noise. EMG noise is random high-frequency noise to an average moving window filter has
    been used as a candidate to remove it. A high-pass filter is applied in cascade as well.
    
    Input: signal-the noisy signal; rate-the sampling frequency.
    
    Output: filtered signal.
    """
    # rate is not used!!
    
    h = np.full((8, ), 1/8)
    filtered = sp.signal.convolve(signal,h)
    #filtered = np.convolve(signal, np.ones(100)/100, mode='same') # We apply an average moving window to remove high-frequency noise.
    high = filter_signal(filtered,0.5, 1000, order=3, filtertype='highpass') # We apply a high-pass filter.
    #high is not used?
    
    return filtered



# An argument could be made for moving this function to the peak detection file
def remove_baseline_wander(ecg, sample_rate, cutoff=0.06):
    """
    The functions removes the signal's baseline.
    Input: data-signal stored in an array; sample_rate: sample rate in which the signal was sampled; cutoff-frequency frequency from which the values will be filtered out.
    Output: corrected signal.
    """
    return filter_signal(data = ecg, cutoff = cutoff, sample_rate = sample_rate, filtertype='notch')




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

