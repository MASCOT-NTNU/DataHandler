"""
EstimatedState module should handle the following requests
- synchronise data according to timestamp
"""

import numpy as np
import pandas as pd


class EstimatedState:

    def __init__(self, datapath: str = None) -> None:
        rawLoc = pd.read_csv(datapath + "EstimatedState.csv", delimiter=', ', header=0, engine='python')
        rawLoc.iloc[:, 0] = np.ceil(rawLoc.iloc[:, 0])
        lat_origin = rawLoc["lat (rad)"].groupby(rawLoc["timestamp"]).mean()
        lon_origin = rawLoc["lon (rad)"].groupby(rawLoc["timestamp"]).mean()
        x_loc = rawLoc["x (m)"].groupby(rawLoc["timestamp"]).mean()
        y_loc = rawLoc["y (m)"].groupby(rawLoc["timestamp"]).mean()
        z_loc = rawLoc["z (m)"].groupby(rawLoc["timestamp"]).mean()
        depth = rawLoc["depth (m)"].groupby(rawLoc["timestamp"]).mean()
        time_loc = rawLoc["timestamp"].groupby(rawLoc["timestamp"]).mean()
        # time_sal = rawSal["timestamp"].groupby(rawSal["timestamp"]).mean()
        # time_temp = rawCTDTemp["timestamp"].groupby(rawCTDTemp["timestamp"]).mean()
        # dataSal = rawSal["value (psu)"].groupby(rawSal["timestamp"]).mean()
        # dataTemp = rawCTDTemp.iloc[:, -1].groupby(rawCTDTemp["timestamp"]).mean()
        pass


if __name__ == "__main__":
    t = EstimatedState()
