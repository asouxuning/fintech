import numpy as np
import pandas as pd
import os
import seq_sample

x_cols = ['close', 'open', 'high', 'low', 
         'turnover', 'volume']
y_cols = ['p_change']

path = 'data/hist/'
files = os.listdir(path)

# get stocks dict
# key: stock code
# value: stock data frame
stocks = {}

for f in files:
  code = f.split('.')[0]
  f = path+f
  df = pd.read_csv(f)
  df.index = pd.to_datetime(df['date'])
  df = df.sort_index(ascending=True)
  df = df[x_cols+y_cols]
  stocks[code] = df

def get_stock_dataset(stock_df,n_feats=30,n_labels=5):
  feats = []
  labels = []
  for x,y in seq_sample.get_feats_labels(stock_df,n_feats,n_labels):
    for code in stocks:
      stock = stocks[code]
      feat = stock[min(x.index):max(x.index)]
      label = stock[min(y.index):max(y.index)]
      if len(x) == len(feat) and len(y) == len(label) :
        # normalized
        feat = (feat-feat.mean())/feat.std()
        feats.append(feat)

        # normalized
        y = (y-y.mean())/y.std()
        labels.append(y)

  return (np.stack(feats), np.stack(labels))

#stock = stocks['000009']
#(feats,labels) = get_stock_dataset(stock)

#for code in stocks:
#  stock = stocks[code]
#
#  (feats,labels) = get_stock_dataset(stock)
#  print("%s: " % code)
#  print("feats size: ", len(feats)), "labels size: ", len(labels)

#feats = []
#labels = []
#for code in seqs:
#  seq = seqs[code]
#  for x,y in seq_sample.get_feats_labels(seq,30,5):
#    x = x[x_cols]
#    x = (x-x.mean())/x.std()
#    feats.append(x)
#
#    y = y[y_cols]
#    y = (y-y.mean())/y.std()
#    labels.append(y)

