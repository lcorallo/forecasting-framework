import numpy as np

class MinMaxTransformer():
    def transform(series):
        series_temp = np.copy(series)
        for i in range(len(series_temp)):
            series_temp[i] = (series_temp[i] - np.min(series))/(np.max(series)-np.min(series))
        return series_temp
  