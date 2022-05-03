import numpy as np

from deeplearning.layers import *
from deeplearning.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        out1, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, cache2 = affine_forward(out1, self.params['W2'], self.params['b2'])

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        loss, grads_sm = softmax_loss(scores, y)
        l2_loss = 0.5 * self.reg * (np.sum(self.params['W1']*self.params['W1']) + np.sum(self.params['W2']*self.params['W2']))
        loss = loss + l2_loss

        #backpropagate to get gradients
        grads_out1, grads['W2'], grads['b2'] = affine_backward(grads_sm, cache2)
        #add regularization
        grads['W2'] += self.reg * self.params['W2']
        grads_X, grads['W1'], grads['b1'] = affine_relu_backward(grads_out1, cache1)
        grads['W1'] += self.reg * self.params['W1']

        return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  nonlinearities, and a softmax loss function. The nonlinearities can be
  chosen from sigmoid, tanh, ReLU and LeakyReLU. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  {affine - [batch norm] - nonlinearity - [dropout]} x (L - 1) - affine - softmax
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10, nonlinearity={'type':'ReLU'},
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - nonlinearity: A dictionary containing the type of nonlinearity to use 
      and its hyperparameters. It must contain the key 'type' and the
      corresponding value is the name of the nonlinearity.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.nonlinearity = nonlinearity['type']
    if self.nonlinearity == 'LeakyReLU':
      self.leak = nonlinearity['leak']
    else:
      self.leak = None
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    # L-1 layers weight init.  First layer has different shapes due to inputs
    # since range has 0 indexing, add one to get 1 indexing
    for layer in range(self.num_layers - 1):
        # first layer uses input_dim and rest of the layers uses last hidden dim
        dim_input = input_dim if layer == 0 else hidden_dims[layer-1]

        self.params['W'+str(layer+1)] = np.random.randn(dim_input, hidden_dims[layer]) * weight_scale
        self.params['b'+str(layer+1)] = np.zeros(hidden_dims[layer])

        if self.use_batchnorm:
            self.params['gamma'+str(layer+1)] = np.ones(hidden_dims[layer])
            self.params['beta'+str(layer+1)] = np.zeros(hidden_dims[layer])

    self.params['W'+str(self.num_layers)] = np.random.randn(hidden_dims[self.num_layers-2], num_classes) * weight_scale
    self.params['b'+str(self.num_layers)] = np.zeros(num_classes)

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      # change python2 xrange to python3 range.
      self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
    # Cast all parameters to the correct datatype
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.
    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    caches = {}
    outs = {}
    dropout_caches = {}

    for layer in range(1, self.num_layers):
        input_matrix = X if layer == 1 else outs['Out_l'+str(layer-1)]
        #Depending on whether or not we use batchnorm and dropout, we pass appropriate parameters to forward function
        if self.use_dropout and self.use_batchnorm:
          out, cache = layer_forward(input_matrix, self.params['W'+str(layer)], self.params['b'+str(layer)], self.nonlinearity, 
                                    self.leak, self.use_batchnorm, self.params['gamma'+str(layer)],
                                    self.params['beta'+str(layer)], self.bn_params[layer-1], self.use_dropout, self.dropout_param)
        elif self.use_batchnorm:
          out, cache = layer_forward(input_matrix, self.params['W'+str(layer)], self.params['b'+str(layer)], self.nonlinearity, 
                                    self.leak, self.use_batchnorm, self.params['gamma'+str(layer)],
                                    self.params['beta'+str(layer)], self.bn_params[layer-1], self.use_dropout)
        elif self.use_dropout:
          out, cache = layer_forward(input_matrix, self.params['W'+str(layer)], self.params['b'+str(layer)], self.nonlinearity, self.leak, 
                                    self.use_batchnorm, use_dropout=self.use_dropout, dropout_param=self.dropout_param)
        else:
          out, cache = layer_forward(input_matrix, self.params['W'+str(layer)], self.params['b'+str(layer)], self.nonlinearity, self.leak, 
                                    use_batchnorm=self.use_batchnorm, use_dropout=self.use_dropout)
        outs.update({'Out_l'+str(layer): out})
        caches.update({'C'+str(layer): cache})

    # last layer
    scores, cache_out = affine_forward(outs['Out_l'+str(self.num_layers-1)],
                                       self.params['W'+str(self.num_layers)],
                                       self.params['b'+str(self.num_layers)])

    outs.update({'Out_l'+str(self.num_layers): scores})
    caches.update({'C'+str(self.num_layers): cache_out})

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    # FC scores -> softmax. Calculate loss and gradients
    loss, sm_grads = softmax_loss(scores, y)

    # add L2 reg to loss
    w_squared = 0
    for layer in range(self.num_layers):
        w_squared += np.sum(self.params['W'+str(layer+1)] * \
                            self.params['W'+str(layer+1)])
    reg_loss = 0.5 * self.reg * w_squared
    loss += reg_loss

    # backprop
    dxs = {}
    for layer in range(self.num_layers, 0, -1):
        if layer == self.num_layers:
            # last layer, backprop the softmax loss gradient
            dx, dw, db = affine_backward(sm_grads, caches['C'+str(layer)])
        else:
            if self.use_batchnorm:
                dx, dw, db, dgamma, dbeta = layer_backward(dx, caches['C'+str(layer)], self.use_batchnorm, self.use_dropout)
                grads.update({'gamma'+str(layer): dgamma})
                grads.update({'beta'+str(layer): dbeta})
            else:
                dx, dw, db = layer_backward(dx, caches['C'+str(layer)], self.use_batchnorm, self.use_dropout)

        # store the gradients and dxs
        grads['W'+str(layer)] = dw
        grads['b'+str(layer)] = db
        dxs['dx_l'+str(layer)] = dx

        # add regularization gradient contribution
        grads['W'+str(layer)] += self.reg * self.params['W'+str(layer)]

    return loss, grads
