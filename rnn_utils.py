import theano
import theano.tensor as T
import numpy as np

#theano.config.exception_verbosity = 'high'
#theano.config.optimizer = 'fast_compile'
#theano.config.compute_test_value = 'warn'

#def elman_rnn_step(x,s,W,U,b,V,c):
#  # next hidden layer
#  t = T.nnet.sigmoid(T.dot(s,W)+T.dot(x,U)+b)
#  # output layer
#  o = T.nnet.sigmoid(T.dot(t,V)+c)
#  # return next step hidden and this step output
#  return (t,o)

def elman_rnn_step(x,s,W,U,b):
  # next state layer
  t = T.nnet.sigmoid(T.dot(s,W)+T.dot(x,U)+b)
  # return next step state
  return t

class RNN():
  def __init__(self,input_dim,state_dim,dtype = theano.config.floatX):
    self.input_dim = input_dim 
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
    state = T.matrix('state', dtype=self.dtype)
    self.state = state
    
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
    params = [W,U,b]
    
    # RNN's feature is a matrix, 
    # a matrix is a sequence of vector,
    # a sequence is a sigle sample for RNN
    seq = T.tensor3('seq', dtype=self.dtype)
    self.seq = seq
    
    # RNN's theano.scan return hidden sequence *ts*  
    ts,_ = theano.scan(fn=self.rnn_step,
                       sequences=[seq],
                       outputs_info=[state],
                       # transfer RNN model parameters
                       # strictly via *non_sequences*
                       non_sequences=params,
                       strict=True)

    # state sequence
    # 输出张量outputs默认为RNN的state sequence
    # 如果需要进一步对state sequence处理才能得到
    # 想要的输出,可继承output函数对state sequence 进行加工
    self.outputs = self.ts = ts

    # 
    self.params = params

    return ts

  def output(self):
    return self.outupts

  def cost(self):
    y_ = self.outputs
    #self.labels = T.tensor_copy(y_)
    mtype = T.TensorType(dtype=y_.dtype, broadcastable=y_.broadcastable)
    self.labels = mtype('labels')
    y = self.labels
    loss = T.sum(-y*T.log(y_)-(1-y)*T.log(1-y_))
    self.loss = loss
    return loss

  # learning rate, default 0.1
  def optimizer(self,lr = 0.1): 
    # gradients
    grads = T.grad(cost=self.loss, wrt=self.params)
    
    # updates params
    updates = [(param,param-lr*grad) for (param,grad) in zip(self.params,grads)]

    self.updates = updates
    return updates

  def compile_train(self):
    train = theano.function(inputs=[self.seq,self.labels,self.state],
			       outputs=[self.outputs],
                               updates = self.updates)
    self.train = train
    return train
    
  def compile_predict(self):
    predict = theano.function(inputs=[self.seq,self.state],
                              outputs=self.outputs)
    self.predict = predict
    return predict
