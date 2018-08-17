import numpy as np
import chainer
from chainer.functions.activation import sigmoid
from chainer import link
from chainer.links.connection import linear
import chainer.functions as F
from chainer import initializers
import math
from chainer import cuda
from chainer.functions.activation import tanh, relu, sigmoid, clipped_relu
from chainer.functions.math import linear_interpolate
import chainer.links as L

###
# Implementation of custom links and layers


###
# Offset link used in Elman layer to learn initial offset

class Offset(link.Link):
    """
    Implementation of offset term to initialize Elman hidden states at t=0
    """

    def __init__(self, n_params, initW=0):

        super(Offset, self).__init__()

        self.add_param('X', (1, n_params), initializer=chainer.initializers.Constant(initW, dtype='float32'))

    def __call__(self, z):
        return F.broadcast_to(self.X, z.shape)

###
# Implementation Elman layer

class Elman(link.Chain):
    """
    Implementation of simple linear Elman layer

    Consider using initW=chainer.initializers.Identity(scale=0.01)
    as in https://arxiv.org/pdf/1504.00941v2.pdf
    (scale=1.0 led to divergence issues in our example)

    """

    def __init__(self, in_size, out_size, initU=None,
                 initW=None, bias_init=0, actfun=relu.relu, maskU=None, maskW=None):
        """

        :param in_size:
        :param out_size:
        :param initU:
        :param initW:
        :param bias_init:
        :param actfun:
        :param maskU: masking of input-hidden weight matrix
        :param maskW: masking of hidden-hidden weight matrix
        """

        super(Elman, self).__init__(
            U=linear.Linear(in_size, out_size,
                            initialW=initU, initial_bias=bias_init),
            W=linear.Linear(out_size, out_size,
                            initialW=initW, nobias=True),
            H0=Offset(out_size),
        )

        self.state_size = out_size
        self.reset_state()
        self.actfun = actfun

        self.maskU = maskU
        self.maskW = maskW

    def to_cpu(self):
        super(Elman, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(Elman, self).to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h):
        assert isinstance(h, chainer.Variable)
        h_ = h
        if self.xp == np:
            h_.to_cpu()
        else:
            h_.to_gpu()
        self.h = h_

    def reset_state(self):
        self.h = None

    def __call__(self, x):

        # add optional mask to input-hidden weights
        if not self.maskU is None:
            if self.U.has_uninitialized_params:
                with cuda.get_device(self.U._device_id):
                    self.U._initialize_params(x.size // x.shape[0])
            self.U.W *= self.maskU

        # add optional mask to hidden-hidden weights
        if not self.maskW is None:
            if self.W.has_uninitialized_params:
                with cuda.get_device(self.W._device_id):
                    self.W._initialize_params(self.h.size // self.h.shape[0])
            self.W.W *= self.maskW


        z = self.U(x)
        if self.h is not None:
            z += self.W(self.h)
        # else:
        #     z += self.H0(z)

        # must be part of layer since the transformed value is part of the
        # representation of the previous hidden state
        self.h = self.actfun(z)

        return self.h


#####
## Dynamic filter implementation of a linear link; can be generalized to all other links (convolutional, LSTM, Elman, ...)

class DynamicFilterLinear(chainer.Link):

    def __init__(self, predictor, in_size, out_size, constantW=True, wscale=1, initialW=None, bias=0, nobias=False, initial_bias=None):
        """

        :param predictor: a neural network which implements the dynamic filter that maps from context inputs to a weight matrix
        :param in_size: size of input
        :param out_size: size of output
        :param constantW: add constant W such that W = C + DF(x)
        :param wscale: scaling factor
        :param initialW: used for weight initialization
        :param bias:
        :param nobias:
        :param initial_bias:
        """

        # Parameters are initialized as a numpy array of given shape.

        super(DynamicFilterLinear, self).__init__()

        self.predictor = predictor
        self.shape = [out_size, in_size]

        self.in_size = in_size
        self.out_size = out_size

        # add bias term
        if nobias:
            self.b = None
        else:
            if initial_bias is None:
                initial_bias = bias
            bias_initializer = initializers._get_initializer(initial_bias)
            self.add_param('b', out_size, initializer=bias_initializer)

        if constantW:

            self._W_initializer = initializers._get_initializer(
                initialW, math.sqrt(wscale))

            if in_size is None:
                self.add_uninitialized_param('C')
            else:
                self._initialize_params(in_size)

        self.constantW = constantW

    def _initialize_params(self, in_size):
        self.add_param('C', (self.out_size, in_size),
                       initializer=self._W_initializer)

    def __call__(self, x, z):
        """

        Args:
            x (~chainer.Variable): Batch of input vectors.
            z (~chainer.Variable): Batch of context vectors.

        Returns:
            ~chainer.Variable: Output of the context layer.

        """

        if self.has_uninitialized_params:
            with cuda.get_device(self._device_id):
                self._initialize_params(x.size // x.shape[0])

        batch_size = x.shape[0]

        # compute adaptive filter
        W = self.predictor(z)

        # reshape linear W to the correct size
        W = F.reshape(W, [batch_size] + self.shape)

        # add constant W if defined
        if self.constantW:
            W += F.tile(self.C,(batch_size,1,1))

        # multiply weights with inputs in batch mode
        y = F.squeeze(F.batch_matmul(W, x), 2)

        # add bias
        y += F.tile(self.b, tuple([batch_size, 1]))

        return y


class GRU_customBase(link.Chain):

    def __init__(self, n_units, n_inputs=None, init=None,
                 inner_init=None, bias_init=0, initOff=0):
        if n_inputs is None:
            n_inputs = n_units

        super(GRU_customBase, self).__init__(
            W_r=linear.Linear(n_inputs, n_units,wscale=0.1,
                              initialW=init, initial_bias=bias_init),
            U_r=linear.Linear(n_units, n_units,wscale=0.1,
                              initialW=inner_init, initial_bias=bias_init),
            W_z=linear.Linear(n_inputs, n_units,wscale=0.1,
                              initialW=init, initial_bias=bias_init),
            U_z=linear.Linear(n_units, n_units,wscale=0.1,
                              initialW=inner_init, initial_bias=bias_init),
            W=linear.Linear(n_inputs, n_units,wscale=0.1,
                            initialW=init, initial_bias=bias_init),
            U=linear.Linear(n_units, n_units,wscale=0.1,
                            initialW=inner_init, initial_bias=bias_init),
            H0=Offset(n_units,initOff),
        )

class GRU_custom(GRU_customBase):
    """Stateful Gated Recurrent Unit function (GRU).

    Stateful GRU function has six parameters :math:`W_r`, :math:`W_z`,
    :math:`W`, :math:`U_r`, :math:`U_z`, and :math:`U`.
    All these parameters are :math:`n \\times n` matrices,
    where :math:`n` is the dimension of hidden vectors.

    Given input vector :math:`x`, Stateful GRU returns the next
    hidden vector :math:`h'` defined as

    .. math::

       r &=& \\sigma(W_r x + U_r h), \\\\
       z &=& \\sigma(W_z x + U_z h), \\\\
       \\bar{h} &=& \\tanh(W x + U (r \\odot h)), \\\\
       h' &=& (1 - z) \\odot h + z \\odot \\bar{h},

    where :math:`h` is current hidden vector.

    As the name indicates, :class:`~chainer.links.StatefulGRU` is *stateful*,
    meaning that it also holds the next hidden vector `h'` as a state.
    Use :class:`~chainer.links.GRU` as a stateless version of GRU.

    Args:
        in_size(int): Dimension of input vector :math:`x`.
        out_size(int): Dimension of hidden vector :math:`h`.
        init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the
            GRU's input units (:math:`W`). Maybe be `None` to use default
            initialization.
        inner_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the GRU's inner
            recurrent units (:math:`U`).
            Maybe be ``None`` to use default initialization.
        bias_init: A callable or scalar used to initialize the bias values for
            both the GRU's inner and input units. Maybe be ``None`` to use
            default initialization.

    Attributes:
        h(~chainer.Variable): Hidden vector that indicates the state of
            :class:`~chainer.links.StatefulGRU`.

    .. seealso:: :class:`~chainer.functions.GRU`

    """

    def __init__(self, in_size, out_size, init= None,
                 inner_init=None, bias_init=0, initOff=0):
        super(GRU_custom, self).__init__(
            out_size, in_size, init, inner_init, bias_init, initOff)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(GRU_custom, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(GRU_custom, self).to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h):
        assert isinstance(h, chainer.Variable)
        h_ = h
        if self.xp == np:
            h_.to_cpu()
        else:
            h_.to_gpu(self._device_id)
        self.h = h_

    def reset_state(self):
        self.h = None

    def __call__(self, x):
        z = self.W_z(x)
        h_bar = self.W(x)
        if self.h is not None:
            r = sigmoid.sigmoid(self.W_r(x) + self.U_r(self.h))
            z += self.U_z(self.h)
            h_bar += self.U(r * self.h)
        else:
            r = sigmoid.sigmoid(self.W_r(x) + self.U_r(self.H0(z)))
            z += self.U_z(self.H0(z))
            h_bar += self.U(r * self.H0(z))

        z = sigmoid.sigmoid(z)
        h_bar = clipped_relu.clipped_relu(h_bar, z=1.0)

        if self.h is not None:
            h_new = linear_interpolate.linear_interpolate(z, h_bar, self.h)
        else:
            h_new = z * h_bar

        self.h = h_new

        return self.h


class GRU_custom2(GRU_customBase):

    def __init__(self, in_size, out_size, init= None,
                 inner_init=None, bias_init=0, initOff=0):
        super(GRU_custom2, self).__init__(
            out_size, in_size, init, inner_init, bias_init, initOff)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(GRU_custom2, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(GRU_custom2, self).to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h):
        assert isinstance(h, chainer.Variable)
        h_ = h
        if self.xp == np:
            h_.to_cpu()
        else:
            h_.to_gpu(self._device_id)
        self.h = h_

    def reset_state(self):
        self.h = None

    def __call__(self, x):
        z = self.W_z(x)
        h_bar = self.W(x)
        if self.h is not None:
            r = sigmoid.sigmoid(self.W_r(x) + self.U_r(self.h))
            z += self.U_z(self.h)
            h_bar += self.U(r * self.h)
        else:
            r = sigmoid.sigmoid(self.W_r(x) + self.U_r(self.H0(z)))
            z += self.U_z(self.H0(z))
            h_bar += self.U(r * self.H0(z))



        z = sigmoid.sigmoid(z)

        h_bar = relu.relu(h_bar)

        if self.h is not None:
            h_new = linear_interpolate.linear_interpolate(z, h_bar, self.h)
        else:
            h_new = z * h_bar

        self.h = h_new

        return self.h

class CRNN(link.Chain):
    """CRNN unit"""



    def __init__(self, n_input, n_out, actfun=relu.relu):
        k = 3 # kernel size
        s = 1 # stride
        p = 1 # padding
        super(CRNN, self).__init__( U=L.Convolution2D(n_input, n_out, k, s, p),
            W=L.Convolution2D(n_out, n_out, k, s, p),
            )
        self.state_size = n_out
        self.actfun=actfun
        self.reset_state()

    def to_cpu(self):
        super(CRNN, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(CRNN, self).to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h):
        assert isinstance(h, chainer.Variable)
        h_ = h
        if self.xp == np:
            h_.to_cpu()
        else:
            h_.to_gpu(self._device_id)
        self.h = h_

    def reset_state(self):
        self.h = None

    def __call__(self, x):



        z = self.U(x)
        if self.h is not None:
            z += self.W(self.h)

        # must be part of layer since the transformed value is part of the
        # representation of the previous hidden state
        self.h = self.actfun(z)
        return self.h
