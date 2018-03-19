from pandas import Series,DataFrame
import pandas as pd
import numpy as np

# 从一个DataFrame表示的时间序列中产生
# 样本的特征和标签
# 时间序列是一个逐次抽取定长子序列的过程

# 在时间序列中
# DataFrame偏head的数据是时间上最新的数据
# 所以遍历过程中,先按次取m个较新的元素 
# 构成label,再接着按次取n个较旧的元素作为features。
# 用较旧的n个元素来预测m个较新的元素
# 样本的个数：
#   序列的长度减去刚好不能构成一个样本的长度
#   在这里序列的长度为len(df)
#   整个序列的长度为len(df)
#   由label和feature构成的样本总长度为m+n
#   所以,刚刚不够构成一个样本的长度为m+n-1
#   进而能够构成样本的个数为len(df)-(m+n-1)
# df: DataFrame
# n: length of a feature 
# m: length of a label
def get_seq_samples(df,n,m):
  feats = []
  labels = []

  for i in range(0,len(df)-(m+n-1)):
    label = df[i:i+m].values
    labels.append(label)
    feat = df[i+m:i+m+n].values
    feats.append(feat)
  
  feats = np.stack(feats,axis=0)
  labels = np.stack(labels,axis=0)
  return feats,labels

# 从头开始循环一个序列,并逐个抽取定长子序列
# seq表示序列,ws表示要抽取子序列的长度,即window size
def loop_sub_seq(seq, ws):
  for i in range(len(seq)-(ws-1)):
    yield seq[i:i+ws]

# 遍历序列获得子序列,
# 并将子序列样本分割成特征和标签
# 元组对予以返回
def get_sample(seq, ws, cut):
  for ss in loop_sub_seq(seq, ws):
    yield (ss[:cut], ss[cut:])

def get_feats_labels(seq, n_feats, n_labels):
  for feat,label in get_sample(seq, n_feats+n_labels, n_feats):
    yield feat,label

#def loop_sub_seqs(seqs, ws):
#  l = min([len(seq) for seq in seqs])
#  for i in range(l-(ws-1)):
#    yield [seq[i:i+ws] for seq in seqs]

# 遍历序列获得子序列,
# 并将子序列样本分割成特征和标签batch
# 元组对予以返回
#def get_batch(seqs, ws, cut):
#  for seq in loop_sub_seqs(seqs,ws): 
#    l = [(ss[:cut],ss[cut:]) for ss in seq]
#    (x,y) = zip(*l)
#    yield (np.stack(x,axis=1),
#           np.stack(y,axis=1))


