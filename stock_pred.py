import rnn_utils
import stock_data
import seq_sample
import numpy as np

path = 'data/hist/'
stocks = stock_data.fetch_group_stocks_from_fs(path)
feats,labels = stock_data.get_stock_sample(stocks, '000009', 30, 5)

(feats,labels) = seq_sample.shufflelists([feats,labels])
feats = seq_sample.get_rnn_batch(feats,128)
labels = seq_sample.get_rnn_batch(labels,128)

input_dim = feats[0].shape[1]
state_dim = 10

rnn = rnn_utils.RNN(input_dim, state_dim)
rnn.forward()
# rnn.output()
rnn.cost()
rnn.backward()
train = rnn.compile_train()
predict = rnn.compile_predict()

#batch_size = 
#initial_state = np.zeros(
