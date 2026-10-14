"""Temporal padding and metric formatting used by the reference workflows."""
import torch

mre = lambda truth, pred: ((truth - pred).pow(2).sum(axis=-1).sqrt() / truth.pow(2).sum(axis=-1).sqrt()).mean()
num2p = lambda value: f"{100 * value:.2f}%"

def Padding(data, lag):
    '''
    Extract time-series of lenght equal to lag from longer time series in data, whose dimension is (number of time series, sequence length, data shape)
    '''
    
    data_out = torch.zeros(data.shape[0] * data.shape[1], lag, data.shape[2])

    for i in range(data.shape[0]):
        for j in range(1, data.shape[1] + 1):
            if j < lag:
                data_out[i * data.shape[1] + j - 1, -j:] = data[i, :j]
            else:
                data_out[i * data.shape[1] + j - 1] = data[i, j - lag : j]

    return data_out
