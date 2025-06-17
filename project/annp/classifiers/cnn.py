from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class DeeperConvNet(object):
    """
    A deeper convolutional network:

    conv - relu - conv - relu - 2x2 maxpool - affine - relu - dropout - affine - softmax
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=(32, 64),
        filter_size=3,
        hidden_dim=128,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        use_batchnorm=True,
        use_dropout=True,
        dropout_keep_ratio=0.5,
        dtype=np.float32,
    ):
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        C, H, W = input_dim
        F1, F2 = num_filters

        # Conv1
        self.params['W1'] = weight_scale * np.random.randn(F1, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(F1)

        # Conv2
        self.params['W2'] = weight_scale * np.random.randn(F2, F1, filter_size, filter_size)
        self.params['b2'] = np.zeros(F2)

        if self.use_batchnorm:
            self.params['gamma1'] = np.ones(F1)
            self.params['beta1'] = np.zeros(F1)
            self.params['gamma2'] = np.ones(F2)
            self.params['beta2'] = np.zeros(F2)

        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        H_pool = H // 2
        W_pool = W // 2
        self.params['W3'] = weight_scale * np.random.randn(F2 * H_pool * W_pool, hidden_dim)
        self.params['b3'] = np.zeros(hidden_dim)

        self.params['W4'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b4'] = np.zeros(num_classes)

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': 1 - dropout_keep_ratio}

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for _ in range(2)]

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        mode = 'test' if y is None else 'train'

        if self.use_dropout:
            self.dropout_param['mode'] = mode

        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

        conv_param = {'stride': 1, 'pad': (W1.shape[2] - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        out, cache = {}, {}

        # conv - relu - (batchnorm) - conv - relu - (batchnorm) - pool
        out['1'], cache['1'] = conv_forward_fast(X, W1, b1, conv_param)
        if self.use_batchnorm:
            out['1_bn'], cache['1_bn'] = spatial_batchnorm_forward(out['1'], self.params['gamma1'], self.params['beta1'], self.bn_params[0])
            out['1_relu'], cache['1_relu'] = relu_forward(out['1_bn'])
        else:
            out['1_relu'], cache['1_relu'] = relu_forward(out['1'])

        out['2'], cache['2'] = conv_forward_fast(out['1_relu'], W2, b2, conv_param)
        if self.use_batchnorm:
            out['2_bn'], cache['2_bn'] = spatial_batchnorm_forward(out['2'], self.params['gamma2'], self.params['beta2'], self.bn_params[1])
            out['2_relu'], cache['2_relu'] = relu_forward(out['2_bn'])
        else:
            out['2_relu'], cache['2_relu'] = relu_forward(out['2'])

        out['pool'], cache['pool'] = max_pool_forward_fast(out['2_relu'], pool_param)

        # affine - relu - dropout
        out['3'], cache['3'] = affine_relu_forward(out['pool'], W3, b3)

        if self.use_dropout:
            out['3_drop'], cache['3_drop'] = dropout_forward(out['3'], self.dropout_param)
            out_fc = out['3_drop']
        else:
            out_fc = out['3']

        # final affine
        scores, cache['4'] = affine_forward(out_fc, W4, b4)

        if mode == 'test':
            return scores

        loss, grads = 0, {}

        # loss
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (
            np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2) + np.sum(W4**2)
        )

        # Backward
        dout, grads['W4'], grads['b4'] = affine_backward(dscores, cache['4'])

        if self.use_dropout:
            dout = dropout_backward(dout, cache['3_drop'])

        dout, grads['W3'], grads['b3'] = affine_relu_backward(dout, cache['3'])
        grads['W3'] += self.reg * W3

        dout = max_pool_backward_fast(dout, cache['pool'])
        dout = relu_backward(dout, cache['2_relu'])

        if self.use_batchnorm:
            dout, grads['gamma2'], grads['beta2'] = spatial_batchnorm_backward(dout, cache['2_bn'])

        dout, grads['W2'], grads['b2'] = conv_backward_fast(dout, cache['2'])
        grads['W2'] += self.reg * W2

        dout = relu_backward(dout, cache['1_relu'])

        if self.use_batchnorm:
            dout, grads['gamma1'], grads['beta1'] = spatial_batchnorm_backward(dout, cache['1_bn'])

        _, grads['W1'], grads['b1'] = conv_backward_fast(dout, cache['1'])
        grads['W1'] += self.reg * W1

        return loss, grads
