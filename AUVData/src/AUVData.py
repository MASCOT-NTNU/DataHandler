"""
This section is used to merge and synchronize data from different sensors according to their associated timestamps.
It utilizes the timestamp from the estimated state in the AUV as the reference time. Then the following temperature and
salinity timestamp are selected using the minimum squared distance among them.
"""
from WGS import WGS
import numpy as np
import pandas as pd
from datetime import datetime
import os
import time
from math import radians, degrees
import matplotlib.pyplot as plt

date_string = "20210617"
datapath = os.getcwd() + "/../../../../Data/Nidelva/" + date_string + "/"

"""
Step I: obtain raw data from spread sheet. 
"""
rawTemp = pd.read_csv(datapath + "Temperature.csv", delimiter=', ', header=0, engine='python', encoding= 'unicode_escape')
rawLoc = pd.read_csv(datapath + "EstimatedState.csv", delimiter=', ', header=0, engine='python', encoding= 'unicode_escape')
rawSal = pd.read_csv(datapath + "Salinity.csv", delimiter=', ', header=0, engine='python', encoding= 'unicode_escape')
rawDepth = pd.read_csv(datapath + "Depth.csv", delimiter=', ', header=0, engine='python', encoding= 'unicode_escape')

"""
Step II: make all timestamps to be integer for later use. It is then easier to group together. 
"""
rawSal.iloc[:, 0] = np.ceil(rawSal.iloc[:, 0])
rawTemp.iloc[:, 0] = np.ceil(rawTemp.iloc[:, 0])
rawCTDTemp = rawTemp[rawTemp.iloc[:, 2] == 'SmartX']
rawLoc.iloc[:, 0] = np.ceil(rawLoc.iloc[:, 0])
rawDepth.iloc[:, 0] = np.ceil(rawDepth.iloc[:, 0])
rawDepth.iloc[:, 0] = np.ceil(rawDepth.iloc[:, 0])

"""
Step III: group together all values close to one timestamp
"""
lat_origin = rawLoc["lat (rad)"].groupby(rawLoc["timestamp"]).mean()
lon_origin = rawLoc["lon (rad)"].groupby(rawLoc["timestamp"]).mean()
x_loc = rawLoc["x (m)"].groupby(rawLoc["timestamp"]).mean()
y_loc = rawLoc["y (m)"].groupby(rawLoc["timestamp"]).mean()
z_loc = rawLoc["z (m)"].groupby(rawLoc["timestamp"]).mean()
depth = rawLoc["depth (m)"].groupby(rawLoc["timestamp"]).mean()
time_loc = rawLoc["timestamp"].groupby(rawLoc["timestamp"]).mean()
time_sal= rawSal["timestamp"].groupby(rawSal["timestamp"]).mean()
time_temp = rawCTDTemp["timestamp"].groupby(rawCTDTemp["timestamp"]).mean()
dataSal = rawSal["value (psu)"].groupby(rawSal["timestamp"]).mean()
dataTemp = rawCTDTemp.iloc[:, -1].groupby(rawCTDTemp["timestamp"]).mean()

"""
Prepration: set origin for WGS.
"""
# lat_o = degrees(np.mean(lat_origin))
# lon_o = degrees(np.mean(lon_origin))
# WGS.set_origin(lat_o, lon_o)

"""
Step IV: synchronize all data together. 
"""
dataset = []
timestamp = []
xauv = []
yauv = []
zauv = []
dauv = []
sal_auv = []
temp_auv = []
lat_auv = []
lon_auv = []

for i in range(len(time_loc)):
    if np.any(time_sal.isin([time_loc.iloc[i]])) and np.any(time_temp.isin([time_loc.iloc[i]])):
        timestamp.append(time_loc.iloc[i])
        xauv.append(x_loc.iloc[i])
        yauv.append(y_loc.iloc[i])
        zauv.append(z_loc.iloc[i])
        dauv.append(depth.iloc[i])
        lat_temp, lon_temp = WGS.xy2latlon(x_loc.iloc[i], y_loc.iloc[i])
        lat_auv.append(lat_temp)
        lon_auv.append(lon_temp)
        sal_auv.append(dataSal[time_sal.isin([time_loc.iloc[i]])].iloc[0])
        temp_auv.append(dataTemp[time_temp.isin([time_loc.iloc[i]])].iloc[0])
    else:
        print(datetime.fromtimestamp(time_loc.iloc[i]))
        continue


# lat4, lon4 = 63.446905, 10.419426  # right bottom corner
# lat_auv = np.array(lat_auv).reshape(-1, 1)
# lon_auv = np.array(lon_auv).reshape(-1, 1)
# Dx = deg2rad(lat_auv - lat4) / 2 / np.pi * circumference
# Dy = deg2rad(lon_auv - lon4) / 2 / np.pi * circumference * np.cos(deg2rad(lat_auv))
#
# xauv = np.array(xauv).reshape(-1, 1)
# yauv = np.array(yauv).reshape(-1, 1)
#
# alpha = deg2rad(60)
# Rc = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
# TT = (Rc @ np.hstack((Dx, Dy)).T).T
# xauv_new = TT[:, 0].reshape(-1, 1)
# yauv_new = TT[:, 1].reshape(-1, 1)
#
# zauv = np.array(zauv).reshape(-1, 1)
# dauv = np.array(dauv).reshape(-1, 1)
# sal_auv = np.array(sal_auv).reshape(-1, 1)
# temp_auv = np.array(temp_auv).reshape(-1, 1)
# timestamp = np.array(timestamp).reshape(-1, 1)
#
# datasheet = np.hstack((timestamp, lat_auv, lon_auv, xauv, yauv, zauv, dauv, sal_auv, temp_auv))
# np.savetxt(os.getcwd() + "data.txt", datasheet, delimiter = ",")

# #%%
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# import scipy.stats as stats
#
# # sal = np.array(sal_auv)
# # plt.plot(sal)
# # plt.show()
# residual = sal - np.mean(sal)
# fig = sm.qqplot(residual, line='45', fit=True)
# plt.show()



# TODO: concatenage all data for all dates
# DO QQPLOt for different layers, different dates, different conditions

sal = np.array(sal_auv)
x = np.array(xauv)
y = np.array(yauv)
z = np.array(zauv)
timestamp = np.array(timestamp)

df = pd.DataFrame(sal, columns=['salinity'])
df.to_csv(date_string + "_salinity.csv", index=False)

#%%
plt.plot(zauv)
plt.plot(dauv)
plt.show()
