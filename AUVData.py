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

# import pyopencl as cl
#
# platform = cl.get_platforms()[0]
# GPUs = platform.get_devices()
# for GPU in GPUs:
#     if "AMD" in GPU.name:
#         break
#
# Test GPU capability
# ctx = cl.Context(devices=[GPU])
# queue = cl.CommandQueue(ctx)


class AUVData:

    def __init__(self, datapath):
        self.load_raw_data(datapath)
        self.organise_data()

    def load_raw_data(self, datapath=None):
        t1 = time.time()
        self.datapath = datapath
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
        self.temp = self.data['Temperature']
        self.temperature = self.temp[self.temp[' entity '] == 73]

        self.lat_origin = np.rad2deg(self.coordinates[' lat (rad)'])
        self.lon_origin = np.rad2deg(self.coordinates[' lon (rad)'])

        self.x = self.coordinates[' x (m)']
        self.y = self.coordinates[' y (m)']
        # self.z = self.coordinates[' z (m)']
        self.depth = self.coordinates[' depth (m)']

    def synchronise_salinity_data(self):
        t1 = time.time()
        self.time_coordinates = self.coordinates.iloc[:, 0].to_numpy()
        self.time_salinity = self.salinity.iloc[:, 0].to_numpy()

        self.dm1 = vectorise(self.time_salinity) @ np.ones([1, len(self.time_coordinates)])
        print("S1: Finished DM1, time consumed: ", time.time() - t1)
        os.system("say Finished data sync 1")
        self.dm2 = np.ones([len(self.time_salinity), 1]) @ vectorise(self.time_coordinates).T
        print("S2: Finished DM2, time consumed: ", time.time() - t1)
        os.system("say Finished data sync 2")
        self.dm = (self.dm1 - self.dm2) ** 2
        print("S3: Finished DM, time consumed: ", time.time() - t1)
        os.system("say Finished data sync 3")
        self.ind_sync = np.argmin(self.dm, axis=1)
        print("S4: Finished ind searching, time consumed: ", time.time() - t1)
        os.system("say Finished data sync all")

        x = self.x[self.ind_sync]
        y = self.y[self.ind_sync]
        depth = self.depth[self.ind_sync]
        lat, lon = xy2latlon(x, y, self.lat_origin[self.ind_sync], self.lon_origin[self.ind_sync])

        dataset = np.vstack((self.time_salinity, lat, lon, depth, self.salinity.iloc[:, -1].to_numpy(),
                             self.temperature.iloc[:, -1])).T
        df = pd.DataFrame(dataset, columns=['timestamp', 'lat', 'lon', 'depth', 'salinity', 'temperature'])
        df.to_csv(self.datapath + "../data_sync.csv", index=False)
        t2 = time.time()
        print("Data is synchronised, time consumed: ", t2 - t1)
        # TODO: add something

    def check_auvdata(self):
        datapath = self.datapath
        # datapath = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/20220511/GOOGLE/RawData/"
        self.load_raw_data(datapath)
        self.organise_data()
        # self.synchronise_salinity_data()


if __name__ == "__main__":
    # datapath = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/20220511/MAFIA/RawData/"
    datapath = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/20220510/"
    ad = AUVData(datapath)
    # ad.synchronise_salinity_data()
    # ad.check_auvdata()




