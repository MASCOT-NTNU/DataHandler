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
import time


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
        self.depth = self.data['Depth']
        self.temperature = self.data['Temperature']
        pass


if __name__ == "__main__":
    # datapath = '/'
    datapath = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/20220511/GOOGLE/"
    ad = AUVData()
    ad.load_raw_data(datapath=datapath)
    ad.organise_data()

#%%
d = {}
if not d:
    print("h");
else: print("f")

import matplotlib.pyplot as plt

plt.plot()
plt.show()
