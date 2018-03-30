import numpy as np
import pandas as pd
import os
import seq_sample


#对一支股的数据进行字段筛选,排序
def clean_stock_data(stock_data):
  cols = ['close', 'open', 'high', 'low', 
          'turnover', 'volume']
  stock_data.index = pd.to_datetime(stock_data['date'])
  stock_data = stock_data[cols]
  stock_data = stock_data.sort_index(ascending=True)
  return stock_data

# get stocks dict
# key: stock code
# value: stock data frame
def fetch_group_stocks_from_fs(path):
  files = os.listdir(path)
  stocks = {}
  for f in files:
    code = f.split('.')[0]
    f = path+f
    df = pd.read_csv(f)
    df = clean_stock_data(df)
    stocks[code] = df
  return stocks

def normalize_seq(x):
  # normalized
  x = (x-x.mean())/x.std()
  return x

# 计算一支股票数据框的收益率
def get_rate_of_return(stock_data):
  s = stock_data['close']
  return (s[-1]-s[0])/s[0]
    
# 从一支股票的数据stock_data中抽取时间序列样本
# stock_data是股票的数据框
def get_stock_samples(stock_data, n_feats=30, n_labels=5):
  feats = []
  labels = []

  # 从选择的股票中提取样本特征feat和样本标签label
  for feat,label in seq_sample.get_feats_labels(stock_data, n_feats, n_labels):
    feat = normalize_seq(feat)
    feats.append(feat)
    labels.append(get_rate_of_return(label))

  feats = np.stack(feats)
  labels = np.stack(labels)
  labels = labels.reshape((labels.shape[0],1))
  
  return (feats,labels)
    
# 在一组股票中预测一支股票的收益率
# stocks是股一组股票的dict
# chosen_code是被预测股票的代码
# 建立一个特征长度为n_feats,标签长度为n_labels的数据的数据集
# 将股票数据逐次提取定长样本序列
# 并分为特征集feats和标签集labels
def get_group_stock_sample(stocks, chosen_code,n_feats=30,n_labels=5):
  chosen_stock_df = stocks[chosen_code]

  feats = []
  labels = []

  # 从选择的股票中提取样本特征x和样本标签y
  for x,y in seq_sample.get_feats_labels(chosen_stock_df,n_feats,n_labels):
    # 从股票组stocks中的其它股票中按x,y的时间来提取样本
    # 并用被选股票的标签y替换掉这些股票的样本标签feat
    for code in stocks:
      #if code == chosen_code:
      #  continue

      stock = stocks[code]
      feat = stock[min(x.index):max(x.index)]
      label = stock[min(y.index):max(y.index)]
      if len(x) == len(feat) and len(y) == len(label) :
        #y = normalize_seq(y)
        feat = normalize_seq(feat)
        feats.append(feat)
        labels.append(get_rate_of_return(y))

  feats = np.stack(feats)
  labels = np.stack(labels)
  labels = labels.reshape((labels.shape[0],1))

  return (feats, labels)

# 将feats和labels重组成适合RNN的样本张量 
def rearange_stock_samples(feats,labels,batch_size):
  (feats,labels) = seq_sample.shufflelists([feats,labels])
  feats = seq_sample.get_seq_batch(feats,batch_size)
  feats = seq_sample.get_rnn_batchs(feats)

  labels = seq_sample.get_seq_batch(labels,batch_size)
  return feats,labels
