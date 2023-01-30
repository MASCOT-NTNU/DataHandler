"""
This class handles SINMOD data operation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-23
"""
import os
import matplotlib.pyplot as plt
from usr_func import *
from numba import vectorize
import multiprocessing as mp
import tkinter as tk
from tkinter import filedialog as fd
SINMOD_MAX_DEPTH_LAYER = 8


# @vectorize(['float32(float32, float32)'])
def get_distance_matrix(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    dx = np.dot(x, np.ones([1, len(y)]))
    dy = np.dot(np.ones([len(x), 1]), y.T)
    return dx - dy


class SINMOD:
    def __init__(self):
        # self.load_sinmod_data()
        pass

    def load_sinmod_data(self, average=True, raw_data=False, save_data=False, filenames=False):
        root = tk.Tk()
        root.withdraw()
        if raw_data:
            if not filenames:
                self.sinmod_files = list(fd.askopenfilenames())
            else:
                self.sinmod_files = filenames
            # TODO add more generalised data handler to deal with different file extensions
            if len(self.sinmod_files) > 0:
                if average:
                    self.salinity_sinmod_average = []
                    t1 = time.time()
                    for file in self.sinmod_files:
                        if file.endswith(".nc"):
                            print(file)
                            ind_before = re.search("samples_", file)
                            ind_after = re.search(".nc", file)
                            self.date_string = file[ind_before.end():ind_after.start()]
                            print(self.date_string)
                            self.sinmod = netCDF4.Dataset(file)
                            ref_timestamp = datetime.strptime(self.date_string, "%Y.%m.%d").timestamp()
                            self.timestamp = np.array(self.sinmod["time"]) * 24 * 3600 + ref_timestamp #change ref timestamp
                            self.lat_sinmod = np.array(self.sinmod['gridLats'])
                            self.lon_sinmod = np.array(self.sinmod['gridLons'])
                            self.depth_sinmod = np.array(self.sinmod['zc'])[:SINMOD_MAX_DEPTH_LAYER]
                            self.salinity_sinmod = np.mean(np.array(self.sinmod['salinity'])[:, :SINMOD_MAX_DEPTH_LAYER, :, :], axis=0)
                            self.salinity_sinmod_average.append(self.salinity_sinmod)
                    t2 = time.time()
                    self.salinity_sinmod_average = np.mean(np.array(self.salinity_sinmod_average), axis=0)
                    print("Data loading time consumed: ", t2 - t1)
                else:
                    print("Here comes not averaging part!")
                    if len(self.sinmod_files) == 1:
                        t1 = time.time()
                        for file in self.sinmod_files:
                            if file.endswith(".nc"):
                                print(file)
                                ind_before = re.search("samples_", file)
                                ind_after = re.search(".nc", file)
                                self.date_string = file[ind_before.end():ind_after.start()]
                                print(self.date_string)
                                self.sinmod = netCDF4.Dataset(file)
                                ref_timestamp = datetime.strptime(self.date_string, "%Y.%m.%d").timestamp()
                                self.timestamp = np.array(
                                    self.sinmod["time"]) * 24 * 3600 + ref_timestamp  # change ref timestamp
                                self.lat_sinmod = np.array(self.sinmod['gridLats'])
                                self.lon_sinmod = np.array(self.sinmod['gridLons'])
                                self.depth_sinmod = np.array(self.sinmod['zc'])[:SINMOD_MAX_DEPTH_LAYER]
                                self.salinity_sinmod = np.array(self.sinmod['salinity'])[:, :SINMOD_MAX_DEPTH_LAYER, :, :]
                        t2 = time.time()
                        print("Data loading time consumed: ", t2 - t1)
                        pass
                    else:
                        raise NotImplementedError("Select only one datafile at each time!")
                        pass
                        # TODO: add not average part
                    pass
            else:
                print(self.sinmod_files)
                pass
            pass
        else:
            # TODO: add request prepared data section
            pass

    def reorganise_sinmod_data(self):
        t1 = time.time()
        self.data_sinmod = []
        for i in range(self.lat_sinmod.shape[0]):
            for j in range(self.lat_sinmod.shape[1]):
                for k in range(len(self.depth_sinmod)):
                    self.data_sinmod.append([self.lat_sinmod[i, j], self.lon_sinmod[i, j],
                                        self.depth_sinmod[k], self.salinity_sinmod_average[k, i, j]])
        self.data_sinmod = np.array(self.data_sinmod)
        t2 = time.time()
        print("Finished data reorganising... Time consumed: ", t2 - t1)

    def get_data_at_coordinates(self, coordinates, filename=False):
        # self.pool = mp.Pool(3)
        print("Start interpolating...")
        self.reorganise_sinmod_data()
        lat_sinmod = self.data_sinmod[:, 0]
        lon_sinmod = self.data_sinmod[:, 1]
        depth_sinmod = self.data_sinmod[:, 2]
        salinity_sinmod = self.data_sinmod[:, 3]

        print("Coordinates shape: ", coordinates.shape)
        self.lat_coordinates = coordinates[:, 0]
        self.lon_coordinates = coordinates[:, 1]
        self.depth_coordinates = coordinates[:, 2]
        ts = time.time()
        x_coordinates, y_coordinates = latlon2xy(self.lat_coordinates, self.lon_coordinates, 0, 0)
        x_sinmod, y_sinmod = latlon2xy(lat_sinmod, lon_sinmod, 0, 0)
        x_coordinates, y_coordinates, depth_coordinates, x_sinmod, y_sinmod, depth_sinmod = \
            map(vectorise, [x_coordinates, y_coordinates, self.depth_coordinates, x_sinmod, y_sinmod, depth_sinmod])
        print("Launching multiprocessing")
        t1 = time.time()
        # dm_x = self.pool.apply_async(get_distance_matrix, args=(x_coordinates, x_sinmod))
        # dm_y = self.pool.apply_async(get_distance_matrix, args=(y_coordinates, y_sinmod))
        # dm_d = self.pool.apply_async(get_distance_matrix, args=(depth_coordinates, depth_sinmod))
        t2 = time.time()
        print("Multiprocess takes: ", t2 - t1)

        t1 = time.time()
        # self.DistanceMatrix_x = dm_x.get()
        self.DistanceMatrix_x = get_distance_matrix(x_coordinates, x_sinmod)
        # self.DistanceMatrix_x = x_coordinates @ np.ones([1, len(x_sinmod)]) - np.ones([len(x_coordinates), 1]) @ x_sinmod.T
        t2 = time.time()
        print("Distance matrix - x finished, time consumed: ", t2 - t1)
        t1 = time.time()
        # self.DistanceMatrix_y = dm_y.get()
        self.DistanceMatrix_y = get_distance_matrix(y_coordinates, y_sinmod)
        # self.DistanceMatrix_y = y_coordinates @ np.ones([1, len(y_sinmod)]) - np.ones([len(y_coordinates), 1]) @ y_sinmod.T
        t2 = time.time()
        print("Distance matrix - y finished, time consumed: ", t2 - t1)
        t1 = time.time()
        # self.DistanceMatrix_depth = dm_d.get()
        self.DistanceMatrix_depth = get_distance_matrix(depth_coordinates, depth_sinmod)
        # self.DistanceMatrix_depth = depth_coordinates @ np.ones([1, len(depth_sinmod)]) - np.ones([len(depth_coordinates), 1]) @ depth_sinmod.T
        t2 = time.time()
        print("Distance matrix - depth finished, time consumed: ", t2 - t1)
        t1 = time.time()
        self.DistanceMatrix = self.DistanceMatrix_x ** 2 + self.DistanceMatrix_y ** 2 + self.DistanceMatrix_depth ** 2
        t2 = time.time()
        print("Distance matrix - total finished, time consumed: ", t2 - t1)
        t1 = time.time()
        self.ind_interpolated = np.argmin(self.DistanceMatrix, axis = 1) # interpolated vectorised indices
        t2 = time.time()
        print("Interpolation finished, time consumed: ", t2 - t1)
        self.salinity_interpolated = vectorise(salinity_sinmod[self.ind_interpolated])
        self.dataset_interpolated = pd.DataFrame(np.hstack((coordinates, self.salinity_interpolated)), columns = ["lat", "lon", "depth", "salinity"])
        t2 = time.time()
        if not filename:
            filename = fd.asksaveasfilename()
        else:
            filename = filename
        self.dataset_interpolated.to_csv(filename, index=False)
        te = time.time()
        print("Data is interpolated successfully! Time consumed: ", te - ts)

    def get_sinmod_at_time_coordinates(self, data=None):
        if data is not None:
            timestamp = data[:, 0]
            lat = data[:, 1]
            lon = data[:, 2]
            depth = data[:, 3]
            pass
        pass

    def check_get_timestamp(self):
        # path_auv = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/202205110/d.csv"
        path_auv = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/20220510/d.csv"
        self.data_auv = pd.read_csv(path_auv).to_numpy()
        pass


if __name__ == "__main__":
    s = SINMOD()
    s.load_sinmod_data(average=False, raw_data=True)
    # s.check_get_timestamp()
    # sinmod.average_all_sinmod_data()

#%%
self = s

timestamp = self.data_auv[:, 0]
lat = self.data_auv[:, 1]
lon = self.data_auv[:, 2]
depth = self.data_auv[:, 3]

DM_time = (vectorise(timestamp) @ np.ones([1, len(self.timestamp)]) -
           np.ones([len(timestamp), 1]) @ vectorise(self.timestamp).T) ** 2

DM_depth = (vectorise(depth) @ np.ones([1, len(self.depth_sinmod)]) -
           np.ones([len(depth), 1]) @ vectorise(self.depth_sinmod).T) ** 2

lat_o = 63.4269097
lon_o = 10.3969375
x_auv, y_auv = latlon2xy(lat, lon, lat_o, lon_o)
x_sinmod, y_sinmod = latlon2xy(self.lat_sinmod.flatten(), self.lon_sinmod.flatten(), lat_o, lon_o)

DM_x = (vectorise(x_auv) @ np.ones([1, len(x_sinmod)]) -
        np.ones([len(x_auv), 1]) @ vectorise(x_sinmod).T) ** 2

DM_y = (vectorise(y_auv) @ np.ones([1, len(y_sinmod)]) -
           np.ones([len(y_auv), 1]) @ vectorise(y_sinmod).T) ** 2

DM_xy = DM_x + DM_y

ind_time = np.argmin(DM_time, axis=1)
ind_depth = np.argmin(DM_depth, axis=1)
ind_d = np.argmin(DM_xy, axis=1)

sal = []
for i in range(len(ind_d)):
    print(i)
    id_t = ind_time[i]
    id_d = ind_depth[i]
    id_xy = ind_d[i]
    sal.append(self.salinity_sinmod[id_t, id_d].flatten()[id_xy])

#%%
s_auv = self.data_auv[:, -2]
r = s_auv - sal
#%%
plt.scatter(y_auv, x_auv, c=r, cmap=get_cmap("BrBG", 10), vmin=-4, vmax=4)
plt.colorbar()
plt.show()
#%%
plt.figure(figsize=(8,8))
plt.plot(sal, s_auv, 'k.')
plt.plot([10, 30], [10, 30], 'r-')
plt.xlabel("SINMOD")
plt.ylabel("AUV")
plt.xlim([10, 30])
plt.ylim([10, 30])
plt.title("AUV v.s. SINMOD")
plt.show()



#%%
# import os
# import numpy as np
# import pyopencl as cl
#
# platform = cl.get_platforms()[0]
# GPUs = platform.get_devices()
# for GPU in GPUs:
#     if "AMD" in GPU.name:
#         break
#
# # Test GPU capability
# context = cl.Context(devices=[GPU])
# queue = cl.CommandQueue(context)
# mf = cl.mem_flags
#
# N = 1000000
# x = np.arange(N).astype(np.float32)
# y = np.arange(N).astype(np.float32)
# dm = np.empty_like(x)
#
# X = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
# Y = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y)
# DM = cl.Buffer(context, mf.WRITE_ONLY, x.nbytes)
#
# program = cl.Program(context, """
# __kernel void DistanceMatrix(__global const float *X, __global const float *Y, __global float *DM)
# {
# int i = get_global_id(0);
# DM[i] = (X[i] - Y[i]) * (X[i] - Y[i]);
# }
# """).build()
#
# program.DistanceMatrix(queue, x.shape, None, X, Y, DM)
# cl._enqueue_read_buffer(queue, DM, dm).wait()
#
# print("finished")
# os.system("say finished")
#
# #%%
# from scipy.spatial import distance_matrix
# distance_matrix()


#%%
# @vectorize(['float32(float32, float32)'])
# def asum(x, y):
#     return x ** 2 + y ** 2
#
# N = 1000000000
# x = np.random.rand(N, 1).astype(np.float32)
# y = np.random.rand(N, 1).astype(np.float32)
# s = asum(x, y)
# t1 = time.time()
# s = asum(x, y)
# t2 = time.time()
# print("Time consumed: ", t2 - t1)
#
#
# t1 = time.time()
# s = x**2 + y**2
# t2 = time.time()
# print("Time consumed: ", t2 - t1)

