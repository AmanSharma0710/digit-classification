from deeplearning.layers import *
from deeplearning.fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def layer_forward(x, w, b, nonlinear, leak=0, use_batchnorm=False, gamma=None, beta=None, bn_param=None, use_dropout=False, dropout_param=None):
    """
    Forward pass for a single layer

    Inputs:
    - x: Input to the layer
    - w, b: Weights for the layer
    - nonlinear: Nonlinearity to apply to the layer
    - leak: Leaky ReLU parameter (if nonlinear is leaky)
    - use_batchnorm: Whether or not to use batch normalization
    - gamma, beta: Batch normalization parameters (if use_batchnorm is True)
    - bn_param: Batch normalization parameters (if use_batchnorm is True)
    - use_dropout: Whether or not to use dropout
    - dropout_param: Dropout parameters (if use_dropout is True)

    Returns a tuple of:
    - out: Output from the layer
    - cache: Object to give to the backward pass
    """
    out, cache = affine_forward(x, w, b)
    if use_batchnorm:
        out, cache = batchnorm_forward(out, gamma, beta, bn_param)
    if nonlinear == 'relu':
        out, cache = relu_forward(out)
    elif nonlinear == 'leaky_relu':
        out, cache = leaky_relu_forward(out, leak)
    elif nonlinear == 'sigmoid':
        out, cache = sigmoid_forward(out)
    elif nonlinear == 'tanh':
        out, cache = tanh_forward(out)
    else:
        raise ValueError('Nonlinearity not recognized')
    if use_dropout:
        out, dropout_cache = dropout_forward(out, dropout_param)
        cache = (cache, dropout_cache)
    return out, cache

def layer_backward(dout, cache, nonlinear, leak=0, use_batchnorm=False, use_dropout=False):
    """
    Backward pass for a single layer

    Inputs:
    - dout: Gradient of the loss with respect to the output of the layer
    - cache: Cache object from the forward pass
    - nonlinear: Nonlinearity to apply to the layer
    - leak: Leaky ReLU parameter (if nonlinear is leaky)
    - use_batchnorm: Whether or not to use batch normalization
    - gamma, beta: Batch normalization parameters (if use_batchnorm is True)
    - bn_param: Batch normalization parameters (if use_batchnorm is True)
    - use_dropout: Whether or not to use dropout
    - dropout_param: Dropout parameters (if use_dropout is True)

    Returns a tuple of:
    - dx: Gradient with respect to the input of the layer
    - dw: Gradient with respect to the weights of the layer
    - db: Gradient with respect to the biases of the layer
    """
    if use_dropout:
        cache, dropout_cache = cache
    else:
        dropout_cache = None
    
    if use_dropout:
        dout = dropout_backward(dout, dropout_cache)

    if nonlinear == 'relu':
        dout = relu_backward(dout, cache)
    elif nonlinear == 'leaky_relu':
        dout = leaky_relu_backward(dout, cache, leak)
    elif nonlinear == 'sigmoid':
        dout = sigmoid_backward(dout, cache)
    elif nonlinear == 'tanh':
        dout = tanh_backward(dout, cache)
    else:
        raise ValueError('Nonlinearity not recognized')
    dgamma, dbeta = None, None
    if use_batchnorm:
        dx, dw, db, dgamma, dbeta = batchnorm_backward(dout, cache)
    
    dx, dw, db = affine_backward(dout, cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
