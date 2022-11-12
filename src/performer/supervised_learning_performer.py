import pandas as pd
import numpy as np

class SlidingWindowPerformer():
  def __init__(self, feature_length=1, target_length=1, target_offset=1):
    self.feature_length = feature_length
    self.target_length = target_length
    self.target_offset = target_offset

  def get(self, series):
    dfTemp = pd.DataFrame({"ref": series})
    dfTemp.ref = dfTemp.ref.astype(float)

    for i in range(self.target_length):
        dfTemp['t+'+str(self.target_length-i)] = dfTemp.ref.shift(i)
  
    dfTemp['t'] = dfTemp.ref.shift(self.target_length)
    for j in range(self.feature_length-1):
        dfTemp['t-'+str(j+1)] = dfTemp.ref.shift(self.target_length+j+1)
    
    dfTemp.drop(columns=["ref"], inplace=True)
    dropLine = self.feature_length+self.target_length-1
    if(self.target_offset > 1):
        dfTemp['t'] = dfTemp['t'].shift(self.target_offset-1)
        for i in range(self.feature_length-1):
            dfTemp['t-'+str(i+1)] = dfTemp['t-'+str(i+1)].shift(self.target_offset-1)
    
    dfTemp.drop(np.arange(0, dropLine+self.target_offset-1), inplace=True)

    dfTemp = dfTemp[dfTemp.columns[::-1]].reset_index().drop(columns=["index"])
    X, Y = dfTemp[dfTemp.columns[0:self.feature_length]].to_numpy().astype(np.float32), dfTemp[dfTemp.columns[self.feature_length:]].to_numpy().astype(np.float32)

    return dfTemp, X, Y