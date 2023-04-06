import torch
import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from math import sqrt

# Define autoencoder model
#class Autoencoder(nn.Module):
#    def __init__(self, input_size, hidden_size):
#        super().__init__()
#        self.encoder = nn.Sequential(
#            nn.Linear(input_size, hidden_size),
#            nn.ReLU()
#        )
#        self.decoder = nn.Sequential(
#            nn.Linear(hidden_size, input_size),
#            nn.Sigmoid()
#        )
#        
#    def forward(self, x):
#        encoded = self.encoder(x)
#        decoded = self.decoder(encoded)
#        return decoded

# Define pulse detection class
class PulseDetector:
    def __init__(self, data, time, num_components=4):
        self.data = data
        self.time = time
        self.num_components = num_components
        
    def detect(self):
        # Normalize data
        data_mean = np.mean(self.data)
        #print(data_mean)
        data_std = np.std(self.data)
        #print(data_std)
        self.data = (self.data - data_mean) / data_std
#
#
        #print(self.data)
        #plt.plot(self.time, self.data)
        #plt.show()

#
        #print(self.data)
        X = self.data.reshape(-1, 1)
        # Fit a Gaussian mixture model to the data
        gmm = GaussianMixture(n_components=self.num_components, max_iter=3000)
        gmm.fit(X)

        peaks = gmm.predict(X)

        
        print(peaks)
        
        # Get the means and variances of the Gaussian components
        means = gmm.means_.flatten()
        variances = gmm.covariances_.flatten()
        
        # Sort the means and variances
        means, variances = zip(*sorted(zip(means, variances)))
        means = np.array(means)
        variances = np.array(variances)
        
        # Compute the threshold for detecting pulses
        #threshold = (means[0] + means[1]) / 2
        std = np.std(variances[0])
        threshold_multiplier = 2
        threshold = (means[1]+means[0])/2 + threshold_multiplier * std
        arr = np.full((len(self.time)), threshold)


        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.time, self.data, label='Data')
        ax.plot(self.time, arr, label='Threshold')

        # Add axis labels and a legend
        ax.set_xlabel('Time')
        ax.set_ylabel('Filtered Value')
        ax.legend()
        plt.plot()
        # Count the number of pulses above the threshold
        num_pulses = len(np.where(self.data > threshold)[0] ) 
        
        # Unnormalize the data and return the result
        
        num_pulses = int(num_pulses * data_std / (means[1] - means[0]))
        return num_pulses
    
    def DetectPeaks(self, threshold=0.2):
        # Normalize data
        data_mean = np.mean(self.data)
        #
        data_std = np.std(self.data)
        #print(data_std)
        self.data = (self.data - data_mean) / data_std

        arr = np.full((len(self.time)), threshold)


        

        peaks1 = (self.data > np.roll(self.data, 1)) & (self.data > np.roll(self.data, -1)) & (self.data > threshold)

        dips1 = (self.data < np.roll(self.data, 1)) & (self.data < np.roll(self.data, -1))

        #Remove peaks below smaller than height 0.05
        peaksanddips = peaks1 | dips1
        nprev = 0
        for (i,n) in enumerate(peaksanddips):
            if(n == True):
                height = self.data[i] - nprev

                
                if(height < 0.05  and height > 0):
                    peaks1[i] = False
                    
                elif(height > -0.05 and height < 0):
                    dips1[i] = False
                    
                nprev = self.data[i]

    
        datafinal = []
        for (i,n) in enumerate(peaks1):
            if(n==True):
                datafinal.append(self.data[i])
            else:
                datafinal.append(0)
            
        #print(datafinal)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.time, self.data, label='Data')
        ax.plot(self.time, arr, label='Threshold')
        ax.plot(self.time, datafinal, label='Peaks')


        ax.set_xlabel('Time')
        ax.set_ylabel('Filtered Value/Treshold')
        ax.legend()
        plt.plot()
        peak_indices = []
        for i in range(0, len(peaks1)):
            if peaks1[i]:
                peak_indices.append(i)
        print(len(peak_indices))

        return peak_indices


