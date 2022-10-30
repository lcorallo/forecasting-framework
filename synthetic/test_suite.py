import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

class TestSuite:
  __REMOTE_URL = "https://raw.githubusercontent.com/lcorallo/forecasting-framework/synthetic_dataset_analysis/synthetic/dataset.csv"
  __SYNTHETIC_SERIES = []
  __RANDOM_SEED_TRAIN_TEST_SET = 1

  def __init__(self):
    tempDataset = pd.read_csv(self.__REMOTE_URL)                                    #Get DataFrame from remote. Dont save it to save RAM
    self.__SYNTHETIC_SERIES = tempDataset.iloc[0].index.to_numpy()[1::].astype(str) #Save index of Synthetic Series


  def __get_test_suite_size__(self):
    return len(self.__SYNTHETIC_SERIES)


  def __get_name_test_series_from_index__(self, index):
    """
      Return the definition of a syntehtic series from the index
    """
    try:
        return self.__SYNTHETIC_SERIES[index]
    except Exception:
        print('Sorry, the index passed as input doesn\'t match any synthetic series')


  def __get_numpy_test_series_from_index__(self, index):
    """
      Return the numpy synthetic test series from specific index
    """
    name_series = self.__get_name_test_series_from_index__(index)
    if(name_series != None):
      tempDataset = pd.read_csv(self.__REMOTE_URL)
      return tempDataset[name_series].str.replace(",",".", regex=True).astype(float).dropna().to_numpy()


  def __get_numpy_test_series_from_name__(self, name_series):
    """
      Return the numpy synthetic test series from specific name series
    """
    tempDataset = pd.read_csv(self.__REMOTE_URL)
    try:
      return tempDataset[name_series].str.replace(",",".", regex=True).astype(float).dropna().to_numpy()
    except Exception:
      print('Sorry, the name series passed as input doesn\'t match any synthetic series')


  def __generate_autoregressive_dataframe(self, series, feature_length=1, target_length=1, offset=1):
    """
      Return pandas.DataFrame auto-regressive supervised-learning table from specific series
    """
    dfTemp = pd.DataFrame({"ref": series})
    dfTemp.ref = dfTemp.ref.astype(float)

    for i in range(target_length):
        dfTemp['t+'+str(target_length-i)] = dfTemp.ref.shift(i)
  
    dfTemp['t'] = dfTemp.ref.shift(target_length)
    for j in range(feature_length-1):
        dfTemp['t-'+str(j+1)] = dfTemp.ref.shift(target_length+j+1)
    
    dfTemp.drop(columns=["ref"], inplace=True)
    dropLine = feature_length+target_length-1
    if(offset > 1):
        dfTemp['t'] = dfTemp['t'].shift(offset-1)
        for i in range(feature_length-1):
            dfTemp['t-'+str(i+1)] = dfTemp['t-'+str(i+1)].shift(offset-1)
    
    dfTemp.drop(np.arange(0, dropLine+offset-1), inplace=True)
    return dfTemp[dfTemp.columns[::-1]].reset_index().drop(columns=["index"])

  def __pre_processing_train_test(self, series, transform, feature_length, target_length, offset):
    """
      Pre-process data in order to create train and test set
    """
    if(transform is not None):
      series = transform(series)
  
    dfTemp = self.__generate_autoregressive_dataframe(series, feature_length, target_length, offset)
    return dfTemp[dfTemp.columns[0:feature_length]].to_numpy().astype(np.float32), dfTemp[dfTemp.columns[feature_length:]].to_numpy().astype(np.float32)


  def __multiple_train_and_test_from_numpy_series__(self, series, feature_length=1, target_length=1, offset=1, transform=None, tensor=False, splits=3):
    """
      Return Multiple Train and Test Set from dataset splits
    """
    X, Y = self.__pre_processing_train_test(series=series, transform=transform, feature_length=feature_length, target_length=target_length, offset=offset)
  
    test_size = int(X.shape[0]/(splits+1))
  
    X_train = []
    X_test = X[-test_size::]
    Y_train = []
    Y_test = Y[-test_size::]
    for i in range(splits):
      train_size = int((i+1) * (test_size) + (X.shape[0] % (splits+1)))

      X_subset = X[-(train_size+test_size)::]
      Y_subset = Y[-(train_size+test_size)::]
    
      X_train.append(X_subset[0:train_size])
      Y_train.append(Y_subset[0:train_size])
      
    if(tensor is True):
      for i in range(splits):
        X_train[i] = torch.tensor(data=X_train[i], dtype=torch.float32)
        X_test[i] = torch.tensor(data=X_test[i], dtype=torch.float32)
        Y_train[i] = torch.tensor(data=Y_train[i], dtype=torch.float32)
        Y_test[i] = torch.tensor(data=Y_test[i], dtype=torch.float32)
        
    return X_train, X_test, Y_train, Y_test


  def __train_and_test_from_numpy_series__(self, series, feature_length=1, target_length=1, offset=1, train_size=0.8, transform=None, mixed=False, tensor=False):
    """
      Return Train and Test Set
    """
    X, Y = self.__pre_processing_train_test(series=series, transform=transform, feature_length=feature_length, target_length=target_length, offset=offset)

    if(tensor is True):
      X = torch.tensor(data=X, dtype=torch.float32)
      Y = torch.tensor(data=Y, dtype=torch.float32)
    
    if(mixed is True):
      return train_test_split(X, Y, train_size=train_size, random_state=self.__RANDOM_SEED_TRAIN_TEST_SET) #X_train, X_test, Y_train, Y_test
    else:
      size = int(len(X)*train_size)
      return X[0:size], X[size::], Y[0:size], Y[size::]