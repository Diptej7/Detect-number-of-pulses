import data_preprocessingSG
import calc_pulses
from matplotlib import pyplot
from sklearn.cluster import DBSCAN
import pandas as pd

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
fig, ax = pyplot.subplots(figsize=(10, 5))
ax.plot(df['_time'], filtered_data_SG, label='SpectralGating')
ax.plot(df['_time'], filtered_data, label='ButterworthFilter')

# Add axis labels and a legend
ax.set_xlabel('Time')
ax.set_ylabel('Filtered Value')
ax.legend()

# Show the plot
pyplot.show()


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Define pulse detection class
class PulseDetector:
    def __init__(self, signal, hidden_size=32, num_epochs=100, lr=0.001):
        self.signal = signal
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.lr = lr
        
    def detect(self):
        # Normalize signal
        signal_norm = (self.signal - np.mean(self.signal)) / np.std(self.signal)
        
        # Convert signal to tensor
        signal_tensor = torch.Tensor(signal_norm).unsqueeze(0)
        
        # Train autoencoder
        model = Autoencoder(input_size=len(signal_norm), hidden_size=self.hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            outputs = model(signal_tensor)
            loss = criterion(outputs, signal_tensor)
            loss.backward()
            optimizer.step()
        
        # Extract latent features from encoder
        encoded = model.encoder(signal_tensor).detach().numpy().squeeze()
        
        # Use KMeans clustering to detect pulses
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(encoded.reshape(-1, 1))
        pulse_labels = kmeans.labels_
        
        # Calculate number of pulses
        num_pulses = np.sum(np.abs(np.diff(pulse_labels)) == 1)
        
        return num_pulses



# Use pulse detector to detect pulses in the signal
pulse_detector = PulseDetector(filtered_data_SG)
detected_num_pulses = pulse_detector.detect()


print("Detected number of pulses: ", detected_num_pulses)

# Plot original signal and reconstructed signal
signal_norm = (filtered_data_SG - np.mean(filtered_data_SG)) / np.std(filtered_data_SG)
signal_tensor = torch.Tensor(signal_norm).unsqueeze(0)
model = Autoencoder(input_size=len(signal_norm), hidden_size=32)
reconstructed = model(signal_tensor).detach().numpy().squeeze()
plt.plot(signal_norm, label='Original Signal')
plt.plot(reconstructed, label='Reconstructed Signal')
plt.legend()
plt.show()
