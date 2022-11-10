import numpy as np

class MinMaxTransformer():
    def transform(series):
        series_temp = np.copy(series)
        for i in range(len(series_temp)):
            series_temp[i] = (series_temp[i] - np.max(series))/(np.max(series)-np.min(series)) + 1
        return series_temp
  