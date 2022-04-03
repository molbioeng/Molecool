import numpy as np



def powerinterference(signal, fs, amplitude='0.3'):
    """
    The function adds noise characterized by 50 or 60 Hz sinusoidal interference, 
    possibly accompanied by a number of harmonics.
    
    Input: signal-clean signal; rate-sampling rate of the signal.
    
    Output: signal with powerinterference added.
    
    """
    x = np.arange(len(signal))
    interference = amplitude*np.sin(2*np.pi*60 * (x/fs))    
    return signal + interference





def emgnoise(signal, mean=5, std=0.05):
    """
    The function generates EMG noise. It does so by adding random noise from a gaussian distribution with a 
    standard deviation of 0.05. Method used from the literature.
    
    Input: signal-the signal we want to add the noise on.
    
    Ouput: the noisy signal with EMG random noise.
    """
    
    noise = np.random.normal(mean,std,len(signal))
    
    # 5 is the mean of the normal distribution you are choosing from
    # 0.05 is the standard deviation of the normal distribution
    return signal + noise




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
    x = np.linspace(0,len(ecg)-1,len(ecg))
#     ecg_bw = ecg + amplitude * np.sin(w*x)
    return ecg + amplitude * np.sin(w*x)




def gen_white_noise(ecg, mean = 0, std=0.05):
    """
    Functions which generates white noise and adds it to the signal.
    
    Inputs: ecg-original signal.
    
    Outputs: ecg with white noise.
    
    """
#     mean = 0
#     std = stan
#     num_samples = len(ecg)
    white_noise = 5.2*np.random.normal(mean, std, size=len(ecg))
#     ecg_wn = ecg + white_noise

    return ecg + white_noise