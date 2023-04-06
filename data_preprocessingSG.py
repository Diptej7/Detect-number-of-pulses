
import pandas
import numpy as np
from datetime import datetime
import scipy.signal as signal
from scipy.signal import stft, istft, butter, filtfilt

class SpectralGating:
    def __init__(self, window_size=2056, hop_size=512, threshold=0.2):
        self.window_size = window_size
        self.hop_size = hop_size
        self.threshold = threshold

    def __call__(self, data):
        #Compute the spectrogram of the input data
        _, _, spec = stft(data, nperseg=self.window_size, noverlap=self.window_size - self.hop_size)

        # Compute the median frequency and sampling rate
        freqs = np.fft.fftfreq(self.window_size)[:self.window_size//2]
        median_freq = np.median(freqs)
        fs = median_freq * 2

        #compute the power spectrum and frequency bins
        power_spec = np.abs(spec) ** 2
        freq_bins = np.arange(spec.shape[0]) * fs / spec.shape[0]

        #compute the time-varying threshold
        threshold = self.threshold * np.mean(power_spec, axis=0)

        #Apply the threshold to the power spectrum
        gated_spec = np.maximum(power_spec - threshold, 0)

        #reconstruct the time-domain signal
        _, filtered_data = istft(np.sqrt(gated_spec) * np.exp(1j * np.angle(spec)), fs=fs, nperseg=self.window_size, noverlap=self.window_size - self.hop_size)

        return filtered_data



class ButterworthFilter:
    def __init__(self, cutoff_freq, sample_rate, order=5):
        nyquist_freq = 0.5 * sample_rate
        normalized_cutoff_freq = cutoff_freq / nyquist_freq
        self.b, self.a = butter(order, normalized_cutoff_freq, btype='lowpass')
    
    def filter(self, data):
        filtered_data = filtfilt(self.b, self.a, data)
        return filtered_data
    
















