import torch
from torch.autograd import Variable
import torch.nn.functional as F

class gumbel:

  def __init__(self, dims):
    self.var = Variable(torch.cuda.FloatTensor(dims),requires_grad=False)
    self.dims = dims
    
  def sample_gumbel(self,shape, eps=1e-20): 
    """Sample from Gumbel(0, 1)"""
    if shape == self.dims:
      self.var.data.uniform_(0,1)
      U = self.var
    else:
      U = Variable(torch.cuda.FloatTensor(shape).uniform_(0,1),requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

  def gumbel_softmax_sample(self,logits, temperature): 
    """ Draw a sample from the Gumbel-Softmax distribution"""
    shape = logits.size()
    g = self.sample_gumbel(shape)
    y = logits + g 
    return F.softmax( y / temperature)

  def gumbel_softmax(self,logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = self.gumbel_softmax_sample(logits, temperature)
    '''
    if hard:
      k = tf.shape(logits)[-1]
      #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
      y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
      y = tf.stop_gradient(y_hard - y) + y
    '''
    return y

