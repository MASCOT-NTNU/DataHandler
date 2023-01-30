"""
Temperature module should handle the following requests
- synchronise data according to timestamp
"""
import numpy as np
import pandas as pd


class Temperature:

    def __init__(self, datapath: str = None) -> None:
        rawTemp = pd.read_csv(datapath + "Temperature.csv", delimiter=', ', header=0, engine='python')
        rawTemp.iloc[:, 0] = np.ceil(rawTemp.iloc[:, 0])
        rawCTDTemp = rawTemp[rawTemp.iloc[:, 2] == 'SmartX']
        time_temp = rawCTDTemp["timestamp"].groupby(rawCTDTemp["timestamp"]).mean()
        dataTemp = rawCTDTemp.iloc[:, -1].groupby(rawCTDTemp["timestamp"]).mean()
        pass


if __name__ == "__main__":
    t = Temperature(None)



