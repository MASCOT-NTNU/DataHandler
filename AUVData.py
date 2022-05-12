"""
This class handles AUV data operation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-05-12
"""
import os
import re
import warnings
import pandas as pd
import numpy as np
import time
from usr_func import xy2latlon, vectorise
from datetime import datetime
import matplotlib.pyplot as plt



class AUVData:

    def __init__(self):
        pass

    def load_raw_data(self, datapath=None):
        t1 = time.time()
        filenames = os.listdir(datapath)
        self.data = dict()
        for file in filenames:
            if file.endswith(".csv"):
                ind_end = re.search(".csv", file)
                variable = file[:ind_end.start()]
                self.data[variable] = pd.read_csv(datapath + file)
        if not self.data:
            warnings.warn("Check filepath! No valid data is found!")
        t2 = time.time()
        print("Finished data loading! Time consumed: ", t2 - t1)

    def organise_data(self):
        self.coordinates = self.data['EstimatedState']
        self.salinity = self.data['Salinity']
        self.temperature = self.data['Temperature']

        self.lat_origin = np.rad2deg(self.coordinates[' lat (rad)'].mean())
        self.lon_origin = np.rad2deg(self.coordinates[' lon (rad)'].mean())

        self.x = self.coordinates[' x (m)'].groupby(self.coordinates.iloc[:, 0]).mean()
        self.y = self.coordinates[' y (m)'].groupby(self.coordinates.iloc[:, 0]).mean()
        self.z = self.coordinates[' z (m)'].groupby(self.coordinates.iloc[:, 0]).mean()
        self.depth = self.coordinates[' depth (m)'].groupby(self.coordinates.iloc[:, 0]).mean()

    def synchronise_salinity_data(self):
        t1 = time.time()
        self.time_coordinates = self.coordinates.iloc[:, 0].to_numpy()
        self.time_salinity = self.salinity.iloc[:, 0].to_numpy()

        self.data_sync = []
        dm = (vectorise(self.time_salinity) @ np.ones([1, len(self.time_coordinates)]) -
              np.ones([len(self.time_salinity), 1]) @ vectorise(self.time_coordinates).T)
        self.ind_sync = np.argmin(dm, axis=1)
        print(self.ind_sync)
        # for i in range(len(self.time_coordinates)):
        #     if (np.any(self.time_salinity.isin([self.time_coordinates.iloc[i]])) and
        #             np.any(self.time_temperature.isin(self.time_coordinates.iloc[i]))):
        #         lat, lon = xy2latlon(self.x.iloc[i], self.y.iloc[i], self.lat_origin, self.lon_origin)
        #         self.data_sync.append([self.time_coordinates.iloc[i], lat, lon, self.z.iloc[i], self.salinity.iloc[i]])
        #     else:
        #         print(datetime.fromtimestamp(self.time_coordinates.iloc[i]))
        #         continue
        t2 = time.time()
        print("Data is synchronised, time consumed: ",t2 - t1)




if __name__ == "__main__":
    # datapath = '/'
    datapath = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/20220511/GOOGLE/"
    ad = AUVData()
    ad.load_raw_data(datapath=datapath)
    ad.organise_data()
    ad.synchronise_salinity_data()

#%%

plt.plot(ad.time_temperature)
plt.show()


