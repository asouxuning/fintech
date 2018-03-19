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

chosen_code = '000009'
chosen_stock = stocks[chosen_code]

feats = []
labels = []
for x,y in seq_sample.get_feats_labels(chosen_stock,30,5):
  for code in stocks:
    stock = stocks[code]
    feat = stock[min(x.index):max(x.index)]
    label = stock[min(y.index):max(y.index)]
    if len(x) == len(feat) and len(y) == len(label) :
      feats.append(feat)
      labels.append(y)


    
    
    
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

