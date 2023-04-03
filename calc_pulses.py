import torch
import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn

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
    def __init__(self, data, num_components=2):
        self.data = data
        self.num_components = num_components
        
    def detect(self):
        # Normalize data
        data_mean = np.mean(self.data)
        data_std = np.std(self.data)
        self.data = (self.data - data_mean) / data_std
        
        # Fit a Gaussian mixture model to the data
        gmm = GaussianMixture(n_components=self.num_components, max_iter=1000)
        gmm.fit(self.data.reshape(-1, 1))
        
        # Get the means and variances of the Gaussian components
        means = gmm.means_.flatten()
        variances = gmm.covariances_.flatten()
        
        # Sort the means and variances
        means, variances = zip(*sorted(zip(means, variances)))
        means = np.array(means)
        variances = np.array(variances)
        
        # Compute the threshold for detecting pulses
        threshold = (means[0] + means[1]) / 2
        
        # Count the number of pulses above the threshold
        num_pulses = len(np.where(self.data > threshold)[0])
        
        # Unnormalize the data and return the result
        num_pulses = int(num_pulses * data_std / (means[1] - means[0]))
        return num_pulses
