import numpy as np
import pandas as pd
import math

class MinMaxTransformer():
    def transform(series):
        series_temp = np.copy(series)
        for i in range(len(series_temp)):
            series_temp[i] = (series_temp[i] - np.min(series))/(np.max(series)-np.min(series))
        return series_temp

class DataFrameToSeriesTransformer():
    def transform(table):
        size = table.shape[1]
        table_temp = np.copy(table)
        none_row = [None] * size
        for i in range(size):
            table_temp = np.append(table_temp, none_row)
        table_temp = table.reshape(-1, size)
        new_table = pd.DataFrame(table_temp)
        for i in range(size):
            for j in range(i+1):
                new_table[size-j-1] = new_table[size-j-1].shift(1)

        list = []
        for i in range(new_table.shape[0]):
            values= [j for j in new_table.iloc[i].to_numpy() if j is not None and not math.isnan(j)]
            if(len(values) > 0): list.append(np.mean(values))

        return np.array(list)