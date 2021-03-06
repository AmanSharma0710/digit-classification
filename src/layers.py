import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an (fully-connected) layer.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = (x.reshape(x.shape[0], -1)).dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = np.array(dout, copy=True)
    dx[x<=0] = 0
    return dx
    
def leaky_relu_forward(x, l):
    """
    Computes the forward pass for a layer of leaky rectified linear units (Leaky-ReLUs).

    Input:
    - x: Inputs, of any shape
    -l: Gradient for the negative terms

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(0, x) + l*np.minimum(0, x)
    cache = (x, l)
    return out, cache


def leaky_relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of leaky rectified linear units (Leaky-ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, (x, l) = None, cache
    dx = np.array(dout, copy=True)
    dx[x<=0] *= l
    return dx

def tanh_forward(x):
    """
    Computes the forward pass for a layer of tanh units.

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.tanh(x)
    cache = (x, out)
    return out, cache

def tanh_backward(dout, cache):
    """
    Computes the backward pass for a layer of tanh units.

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, (x, out) = None, cache
    dx = (1 - out**2) * dout
    return dx

def sigmoid_forward(x):
    """
    Computes the forward pass for a layer of sigmoid units.

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = 1 / (1 + np.exp(-x))
    cache = (x, out)
    return out, cache

def sigmoid_backward(dout, cache):
    """
    Computes the backward pass for a layer of sigmoid units.

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, (x, out) = None, cache
    dx = (1 - out) * out * dout
    return dx

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        sample_mean = x.mean(axis=0)
        sample_var = x.var(axis=0)

        # normalize
        x_norm = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_norm + beta

        # update running mean and variance.
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        cache = (x, sample_mean, sample_var, eps, x_norm, gamma, beta)
    elif mode == 'test':
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    x, mean, var, eps, x_norm, gamma, beta = cache
    dbeta = dout.sum(axis=0)
    dgammax = dout
    dgamma = np.sum(dgammax * x_norm, axis=0)
    dx_norm = dout * gamma
    d_x_minus_mean1 = dx_norm * (var + eps) ** (-1/2)
    d_inv_std = np.sum(dx_norm * (x - mean), axis=0)
    dvar = d_inv_std * (-1/2) * (var + eps) ** (-3/2)
    dsum_sq = np.ones_like(x)/x.shape[0] * dvar
    d_x_minus_mean2  = 2 * (x - mean) *  dsum_sq
    dx2 = d_x_minus_mean1 + d_x_minus_mean2
    dmean = -np.sum(d_x_minus_mean1 + d_x_minus_mean2, axis=0)
    dx1 = np.ones_like(x)/x.shape[0] * dmean
    dx = dx1 + dx2
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    x, mean, var, eps, x_norm, gamma, beta = cache

    dbeta = dout.sum(axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    a = x - mean
    b = var + eps
    m = x.shape[0]
    dx = (1. / m) * gamma * (b**(-1. / 2.)) * (m * dout - dout.sum(axis=0) - a * b**(-1) * (dout * a).sum(axis=0))

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout
        and rescale the outputs to have the same mean as at test time.
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        mask = np.random.rand(*x.shape) < p
        out = x * mask
    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param.values()

    _h = 1 + (H + 2 * pad - HH) // stride
    _w = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((N, F, _h, _w))

    # pad only height and width, not batch size or channel. constant_values
    # default is 0
    x_padded = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant')

    for n in range(N):
        for f in range(F):
            # convolve around height and weight while keep n and f dimensions
            for hh in range(_h):
                for ww in range(_w):
                    out[n, f, hh, ww] = np.sum(x_padded[n, :, hh*stride:hh*stride+HH, ww*stride:ww*stride+WW] * w[f, :]) + b[f]
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    x, w, b, conv_param = cache
    stride, pad = conv_param.values()

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, _h, _w = dout.shape

    x_padded = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant')

    dx = np.zeros_like(x)
    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    # db is per filter, so sum over N, W, H
    db = np.sum(dout, axis=(0, 2, 3))

    # loop go through each
    for n in range(N):
        for f in range(F):
            for hh in range(_h):
                for ww in range(_w):
                    # dw = x * dout, per mask
                    dw[f] += x_padded[n, :, hh*stride:hh*stride+HH, ww*stride:ww*stride+WW] * dout[n, f, hh, ww]
                    # dx = w * dout, per mask
                    dx_padded[n, :, hh*stride:hh*stride+HH, ww*stride:ww*stride+WW] += w[f] * dout[n, f, hh, ww]

    # we remove padding from dx_padded to get dx
    dx = dx_padded[:, :, pad:H+pad, pad:W+pad]
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    N, C, H, W = x.shape
    pool_height, pool_width, stride = pool_param.values()
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    out = np.zeros((N, C, H_out, W_out))  
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    out[n, c, i, j] = np.max(x[n, c, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width])
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    x, pool_param = cache
    pool_height, pool_width, stride = pool_param.values()
    N, C, H, W = x.shape
    HH = 1 + (H - pool_height) // stride
    WW = 1 + (W - pool_width) // stride

    dx = np.zeros((N, C, H, W))

    # gradient only flow through the max value of each slice, so need to get a
    # mask of where the max value of slices
    for hh in range(HH):
      for ww in range(WW):
        x_slice = x[:, :, hh*stride:hh*stride+pool_height, ww*stride:ww*stride+pool_width]
        # mask is where x_slice equals to max of x_slice
        # mask shape is N, C, 1, 1
        mask = (x_slice == np.max(x_slice, axis=(2, 3))[:, :, None, None])
        # pass gradient from dout to dx cells, filter by mask
        dx[:, :, hh*stride:hh*stride+pool_height, ww*stride:ww*stride+pool_width] += (dout[:, :, hh, ww])[:, :, None, None] * mask
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    N, C, H, W = x.shape
    x_transformed = x.transpose(0, 3, 2, 1).reshape(N*H*W, C)
    out, cache = batchnorm_forward(x_transformed, gamma, beta, bn_param)
    out = out.reshape(N, W, H, C).transpose(0, 3, 2, 1)

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None
    N, C, H, W = dout.shape
    dout_reshape = dout.transpose(0, 3, 2, 1).reshape(N*W*H, C)
    dx, dgamma, dbeta = batchnorm_backward_alt(dout_reshape, cache)
    dx = dx.reshape(N, W, H, C).transpose(0, 3, 2, 1)

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C
`
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
