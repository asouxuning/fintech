import rnn_utils
import stock_data
import seq_sample
import numpy as np
import pandas as pd

# 1,data cleaning
#path = 'data/hist/'
#stocks = stock_data.fetch_group_stocks_from_fs(path)
#feats,labels = stock_data.get_stock_sample(stocks, '000009', 30, 5)

# 1,data cleaning
path = 'data/hist/000009.csv'
sd = pd.read_csv(path)
sd = stock_data.clean_stock_data(sd)
(feats,labels) = stock_data.get_stock_samples(sd)


feats_,labels_ = stock_data.rearange_stock_samples(feats,labels,128)
# 2,model design
input_dim,output_dim,batch_size = rnn_utils.get_samples_info(feats_,labels_)
# 经主成分分析,第一主成分和第二主成分包含了信号的
# 97%以上的能量,所以选状态向量维数为2
state_dim = 2

rnn = rnn_utils.RNN_seq2_m1to1(input_dim, output_dim, state_dim, batch_size)
rnn.forward()

# 重写了self.output函数后,更改了输出outputs
rnn.output()

rnn.cost()
rnn.backward()

rnn.compile_train()
rnn.compile_predict()

n_epoches = 100
cost_list = []
for j in range(n_epoches):
  feats_,labels_ = stock_data.rearange_stock_samples(feats,labels,128)
  loss_list = []
  n_samples = len(feats_)
  for i in range(n_samples): 
    loss_ = rnn.train(feats_[i],labels_[i])
    loss_list.append(loss_)
  cost = sum(loss_list)/n_samples
  print("%d:\t%f" % (j,cost))
  cost_list.append(cost)

