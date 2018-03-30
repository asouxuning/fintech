import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

cols=['close', 'open', 'high', 'low', 'turnover', 'volume'] 
def clean_stock_data(data, n_days=5, cols=cols):
  data.index = pd.to_datetime(data['date'])
  data = data[cols]
  data = data.sort_index(ascending=True)

  # 将第n_days天的数据移到当前(第0天)
  # 将第1天的数据移到当前(第0天)
  # 并将它们进行运算
  # pandas的shift函数参數为正时,意味着将当下的数据向未来移(推)
  # pandas的shift函数参数为负时,意味着将未来的数据往当下移(拉)
  data.loc[:,'return'] = data['close'].shift(-n_days) / data['open'].shift(-1) - 1.0

  # 在当前的数据框中将缺失数据NaN都删除
  data.dropna(inplace=True)

  return data 

def make_stock_seq_samples(data, seq_len=30, is_scale=True):
  feats = []
  labels = []
  
  for i in range(len(data)-(seq_len-1)):
    feat = data[cols][i:i+seq_len]
    if is_scale == True:
      feat = scale(feat)
      
    label = data['return'][i+seq_len-1]

    feats.append(feat)
    labels.append(label)

  feats = np.stack(feats)
  labels = np.stack(labels)
  labels = labels.reshape((labels.shape[0],1))
    
  return (feats,labels)

path = 'data/hist/000009.csv'
data = pd.read_csv(path)
data = clean_stock_data(data)
(feats,labels) = make_stock_seq_samples(data)


