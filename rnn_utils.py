import theano
import theano.tensor as T
import numpy as np

def get_samples_info(feats,labels):
  # 根据样本的batch size来决定RNN state的shape
  # 也就是(batch_size,state_dim)
  assert feats.shape[0] == labels.shape[0]

  batch_size = feats.shape[2]
  # 数据样本特征张量最后一阶的维数就是输入特征向量的维数
  input_dim = feats[0].shape[-1] 
  # 数据样本标签张量的最后一阶维数就是输出向量的维数
  output_dim = labels[0].shape[-1]

  return (input_dim,output_dim,batch_size)

def elman_rnn_step(x,s,W,U,b):
  # next state layer
  t = T.nnet.sigmoid(T.dot(s,W)+T.dot(x,U)+b)
  # return next step state
  return t

class RNN():
  def __init__(self,input_dim,output_dim,state_dim,batch_size,dtype=theano.config.floatX): 
    self.input_dim = input_dim    
    self.output_dim = output_dim    
    self.batch_size = batch_size    
    
    self.state_dim = state_dim
    self.rnn_step = elman_rnn_step
    self.dtype = dtype

  # default rnn step is elman rnn
  def set_rnn_step(self,rnn_step):
    self.rnn_step = rnn_step

  # return state sequence 
  def forward(self):
    # intial state
    # transfered to theano.scan outputs_info
    #state = T.matrix('state', dtype=self.dtype)
    self.S = theano.shared(np.zeros((self.batch_size, self.state_dim), dtype=self.dtype))
    #self.state = state
    
    # t-1 hidden layer to t hidden layer transformation matrix
    W_value = np.random.normal(loc=0.0, scale=1.0, 
                               size=(self.state_dim,self.state_dim))
    W_value = W_value.astype(dtype=self.dtype)
    W = theano.shared(value=W_value, name='W')
    
    # input layer to hidden layer transformation matrix
    U_value = np.random.normal(loc=0.0, scale=1.0, 
                               size=(self.input_dim,self.state_dim))
    U_value = U_value.astype(dtype=self.dtype)
    U = theano.shared(value=U_value, name='U')
    
    # bias
    b_value = np.zeros((self.state_dim,),dtype=self.dtype)
    b = theano.shared(value=b_value, name='b')
    
    # RNN model parameters list
    # used for theano.scan non_sequences and T.grad wrt
    self.params = [W,U,b]
    
    # RNN's feature is a matrix, 
    # a matrix is a sequence of vector,
    # a sequence is a sigle sample for RNN
    self.X = T.tensor3('X', dtype=self.dtype)
    
    # RNN's theano.scan return hidden sequence *ts*  
    ts,_ = theano.scan(fn=self.rnn_step,
                       sequences=[self.X],
                       outputs_info=[self.S],
                       # transfer RNN model parameters
                       # strictly via *non_sequences*
                       non_sequences=self.params,
                       strict=True)

    # state sequence
    # 输出张量outputs默认为RNN的state sequence
    # 如果需要进一步对state sequence处理才能得到
    # 想要的输出,可继承output函数对state sequence 进行加工
    self.outputs = self.ts = ts

    return ts

  def output(self):
    return self.outputs

  def cost(self):
    Y_ = self.outputs

    # 复制一个与前向传播结果Y_同阶的符号张量Y
    # Y则作为样本标签输入的占位符
    mtype = T.TensorType(dtype=Y_.dtype, broadcastable=Y_.broadcastable)
    self.Y = mtype('Y')
    Y = self.Y
    #self.loss = T.sum(-Y*T.log(Y_)-(1-Y)*T.log(1-Y_))
    self.loss = T.sum((Y-Y_)**2) / self.batch_size
    return self.loss

  # learning rate, default 0.1
  def backward(self,lr = 0.1): 
    # gradients
    grads = T.grad(cost=self.loss, wrt=self.params)
    
    # updates params
    self.updates = [(param,param-lr*grad) for (param,grad) in zip(self.params,grads)]
    return self.updates

  def compile_train(self):
    train_inner = theano.function(inputs=[self.X,self.Y],
			          outputs=self.loss,
                                  updates = self.updates)
                               
    self.train_inner = train_inner
    return train_inner
    
  def compile_predict(self):
    self.predict_inner = theano.function(inputs=[self.X], outputs=self.outputs)
    return self.predict_inner

  def train(self,feats,labels):
    return self.train_inner(feats,labels)

  def predict(self,inputs):
    return self.predict_inner(inputs)

# 类RNN_seq2scalar继承RNN类,重写了output方法.
# 用于将一个序列seq映射为一个标量值scalar的输出
# 激活函数act将标量映射为一个想要映射的空间
# 如要得到概率,则用simoid
# 如要预测涨跌幅,则用tanh
# 默认的激活函数是恒等函数
class RNN_seq2scalar(RNN):
  def output(self, act=None):
    # hidden layer to output layer transormation matrix 
    V_value = np.random.normal(loc=0.0, scale=1.0,
                               size=(self.state_dim, self.output_dim))
    V_value = V_value.astype(dtype=self.dtype)
    V = theano.shared(value=V_value, name='V')

    # bias
    c_value = np.zeros((self.output_dim,),dtype=self.dtype)
    c = theano.shared(value=c_value, name='c')

    o = self.outputs[-1]
    z = T.dot(o, V)+c
    self.outputs = act(z) if act is not None else z

    # 将新的模型参数V和c加入参数列表
    self.params.append(V)
    self.params.append(c)

    return self.outputs
    
class RNN_seq2prob(RNN_seq2scalar):
  def output(self):
    return RNN_seq2scalar.output(self,act=T.sigmoid)

class RNN_seq2_m1to1(RNN_seq2scalar):
  def output(self):
    return RNN_seq2scalar.output(self,act=T.tanh)
