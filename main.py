import data_preprocessingSG
import calc_pulses
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

df = pd.read_csv('data_signal.csv')
data = df['_value'].values


#df.plot()
#plt.show()

spectral_gating = data_preprocessingSG.SpectralGating()


filtered_data_SG = spectral_gating(data)
filtered_data_SG = filtered_data_SG[:len(data)]
#print(len(filtered_data_SG))

df_filtered_SG = pd.DataFrame({'_time': df['_time'], '_value': filtered_data_SG})
#df_filtered_SG.plot()
#pyplot.show()
#df_filtered.to_csv('output_data.csv', index=False)

# convert '_time' column to datetime
df['_time'] = pd.to_datetime(df['_time'])

# calculate the sampling rate
sampling_rate = 1 / (df['_time'].diff().mean().total_seconds())

# set the cutoff frequency for filtering
cutoff_freq = 0.005 * sampling_rate

# initialize the Butterworth filter
filter = data_preprocessingSG.ButterworthFilter(cutoff_freq, sampling_rate)

# apply the filter to the '_value' column
filtered_data = filter.filter(df['_value'])
t = df['_time']

df_filtered_BWF = pd.DataFrame({'_time': df['_time'], '_value': filtered_data})


#print(df_filtered_BWF)


# Calculate number of pulses

pulse_detector1 = calc_pulses.PulseDetector(filtered_data_SG,t)
num_pulses = len(pulse_detector1.DetectPeaks())
print("Number of pulses(spectral gating): ", num_pulses)


pulse_detector2 = calc_pulses.PulseDetector(filtered_data,t)
num_pulses = len(pulse_detector2.DetectPeaks())
print("Number of pulses(butterworth filter): ", num_pulses)


#peaks = pulse_detector1.DetectPeaks()
#print(len(peaks))


# Create a plot with both filtered signals
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['_time'], filtered_data_SG, label='SpectralGating')
ax.plot(df['_time'], filtered_data, label='ButterworthFilter')

# Add axis labels and a legend
ax.set_xlabel('Time')
ax.set_ylabel('Filtered Value')
ax.legend()

# Show the plot
plt.show()




