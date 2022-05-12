"""
This class handles SINMOD data operation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-23
"""
import os
from tkinter import filedialog as fd
from usr_func import *


class SINMOD:
    def __init__(self):
        self.load_sinmod_data()
        pass

    def load_sinmod_data(self, average=True, raw_data=False, save_data=False, filenames=False):
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
                    print("Time consumed: ", t2 - t1)
                else:
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
        t1 = time.time()
        self.DistanceMatrix_x = x_coordinates @ np.ones([1, len(x_sinmod)]) - np.ones([len(x_coordinates), 1]) @ x_sinmod.T
        t2 = time.time()
        print("Distance matrix - x finished, time consumed: ", t2 - t1)
        t1 = time.time()
        self.DistanceMatrix_y = y_coordinates @ np.ones([1, len(y_sinmod)]) - np.ones([len(y_coordinates), 1]) @ y_sinmod.T
        t2 = time.time()
        print("Distance matrix - y finished, time consumed: ", t2 - t1)
        t1 = time.time()
        self.DistanceMatrix_depth = depth_coordinates @ np.ones([1, len(depth_sinmod)]) - np.ones([len(depth_coordinates), 1]) @ depth_sinmod.T
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


if __name__ == "__main__":
    sinmod = SINMOD()
    # sinmod.average_all_sinmod_data()





