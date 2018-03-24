import numpy as np
import pandas as pd
import os
import seq_sample

cols = ['close', 'open', 'high', 'low', 
         'turnover', 'volume']

def clean_stock_data(df):
  df.index = pd.to_datetime(df['date'])
  df = df.sort_index(ascending=True)
  df = df[cols]
  return df

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

# 计算样本标签的收益率
def get_rate_of_return(label):
  s = label['close']
  return (s[-1]-s[0])/s[0]

#def get_rate_of_returns(labels):
#  rrs = []
#  for i in range(len(labels)):
#    label = labels[i]
#    rr = get_rate_of_return(label)
#    #tr = test_rate_of_return(label)
#    rrs.append(rr)
#  return rrs

#def test_rate_of_return(label):
#  s = label['p_change'] / 100.0
#  p = 1.0#+s[0]
#  for r in s[1:None]:
#    p = (1+r)*p
#  return p - 1.0
    
# 在一组股票中预测一支股票的收益率
# stocks是股一组股票的dict
# chosen_code是被预测股票的代码
# 建立一个特征长度为n_feats,标签长度为n_labels的数据的数据集
# 将股票数据逐次提取定长样本序列
# 并分为特征集feats和标签集labels
def get_stock_sample(stocks, chosen_code,n_feats=30,n_labels=5):
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
        feat = normalize_seq(feat)
        #y = normalize_seq(y)

        feats.append(feat)
        labels.append(get_rate_of_return(y))

  feats = np.stack(feats)
  labels = np.stack(labels)
  labels = labels.reshape(labels.shape[0],1,1)

  return (feats, labels)

#path = 'data/hist/'
#stocks = fetch_group_stocks_from_fs(path)
#feats,labels = get_stock_sample(stocks, '000009', 30, 5)


