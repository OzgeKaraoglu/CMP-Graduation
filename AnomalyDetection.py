from __future__ import division
from matplotlib import pyplot as plt
from numpy import loadtxt
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import collections

data = loadtxt("dataset/heartrate.txt", float)
data_as_frame = pd.DataFrame(data, columns=['Time', 'HeartRate'])
data_as_frame.head()


def kalman_filter(data, kf):
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    kf = kf.em(data,n_iter=20)
    (filtered_state_means, filterd_state_covariances) = kf.filter(data)
    data = filtered_state_means.reshape(-1)
    return data



def anomalies(y, avg, window_size, sigma=1.96):
    residual = y - avg
    std = np.std(residual)
    od = collections.OrderedDict()

    for i, y_i in y.items():
        if (y[i] > avg[i] + (sigma * std)) | (y[i] < avg[i] - (sigma * std)):
            od[i] = y[i]
    return od



type = raw_input("Enter the type: kalman or moving\n")

heartrate = data_as_frame['HeartRate']
time = data_as_frame['Time']
window_size = 75
z_score = 1.96
plt.figure(figsize=(15, 8))
plt.plot(time, heartrate, color="darkgreen")
plt.xlim(0, 1000)
plt.xlabel("Time" , color = 'darkgreen')
plt.ylabel("HeartRate", color = 'darkgreen')


if type == "kalman":
    plt.title("Anomaly Detection with Kalman Filter")
    events = anomalies(heartrate, kalman_filter(heartrate,window_size), window_size, z_score)
    x_anomaly = np.fromiter(events.iterkeys(), dtype=int, count=len(events))
    y_anomaly = np.fromiter(events.itervalues(), dtype=float, count=len(events))


elif type == "moving":

    plt.title("Anomaly Detection with Exponential Moving Average")
    avg = heartrate.ewm(span=window_size, adjust=False).mean()
    plt.plot(time, avg, color='black')
    events = anomalies(heartrate, heartrate.ewm(span=window_size, adjust=False).mean(), window_size, z_score)
    x_anomaly = np.fromiter(events.iterkeys(), dtype=int, count=len(events))
    y_anomaly = np.fromiter(events.itervalues(), dtype=float, count=len(events))



plt.plot(x_anomaly, y_anomaly, "r*", markersize=10)
plt.grid(True)
plt.show()