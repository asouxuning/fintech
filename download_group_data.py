import tushare as ts
import pandas as pd
import time

#df = ts.get_concept_classified()
#df.to_csv('data/hist/concept_classified.csv')

df = pd.read_csv('data/concept_classified.csv', #converters={'code': str})
                 dtype={'code': str})

df = df[df['c_name']=='特斯拉']
l = len(df)
for i in range(l):
  stock = df.iloc[i]
  code = stock['code']
  store_path = 'data/hist/'+code+'.csv'
  print(store_path)
  #data = ts.get_h_data(code,index=True)
  data = ts.get_hist_data(code)
  if data is not None:
    data.to_csv(store_path)
  time.sleep(30)  



