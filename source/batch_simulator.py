import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd

class BatchSimulator:
    def __init__(self, mean, cov, batch_size=500):
        """
        Initialize the BatchSimulator.
        
        Parameters:
        - mean: list, mean of the multivariate normal distribution
        - cov: list, covariance matrix of the multivariate normal distribution
        - batch_size: int, number of observations to simulate per step
        """
        self.mean = mean
        self.cov = cov
        self.batch_size = batch_size

    def generate(self):
        """
        Generate and return a DataFrame with simulated data.
        
        Returns:
        - pd.DataFrame with columns ['x1', 'x2']
        """
        data = multivariate_normal.rvs(mean=self.mean, cov=self.cov, size=self.batch_size)
        return data
    
    def mean_drift(self):
        """
        Simulate a drift in the data by changing the mean.
        
        Returns:
        - pd.DataFrame with columns ['x1', 'x2']
        """
        new_mean = [self.mean[0] + 1, self.mean[1] + 1]
        data = multivariate_normal.rvs(mean=new_mean, cov=self.cov, size=self.batch_size)
        return data
    
    def generate_stream(self, steps=10, drift_start_at=5, drift_end_at=8):
        """
        Generate a stream of data with a drift.
        
        Returns:
        - pd.DataFrame with columns ['x1', 'x2']
        """
        data = []
        drifts = []
        for _ in range(steps):
            if _ < drift_start_at or _ > drift_end_at:
                data.append(self.generate())
                drifts.append(0)
            else:
                data.append(self.mean_drift())
                drifts.append(1)
        return data, drifts