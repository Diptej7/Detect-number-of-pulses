import data_preprocessingSG
import calc_pulses
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

df = pd.read_csv('data_signal.csv')
data = df['_value'].values

spectral_gating = data_preprocessingSG.SpectralGating()


filtered_data_SG = spectral_gating(data)
filtered_data_SG = filtered_data_SG[:len(data)]
print(len(filtered_data_SG))

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

df_filtered_BWF = pd.DataFrame({'_time': df['_time'], '_value': filtered_data})


print(df_filtered_BWF)


# Calculate number of pulses

pulse_detector = calc_pulses.PulseDetector(data)
num_pulses = pulse_detector.detect()
print("Number of pulses(unfiltered): ", num_pulses)

pulse_detector1 = calc_pulses.PulseDetector(filtered_data_SG)
num_pulses = pulse_detector1.detect()
print("Number of pulses(spectral gating): ", num_pulses)


pulse_detector2 = calc_pulses.PulseDetector(filtered_data)
num_pulses = pulse_detector2.detect()
print("Number of pulses(butterworth filter): ", num_pulses)



# Create a plot with both filtered signals
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['_time'], filtered_data_SG, label='SpectralGating')
ax.plot(df['_time'], filtered_data, label='ButterworthFilter')

# Use pulse detector to detect pulses in the signal
pulse_detector = calc_pulses.PulseDetector(filtered_data_SG)
detected_num_pulses = pulse_detector.detect()

# Add axis labels and a legend
ax.set_xlabel('Time')
ax.set_ylabel('Filtered Value')
ax.legend()

# Show the plot
plt.show()
