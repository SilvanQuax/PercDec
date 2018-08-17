from chainer import Chain, ChainList
import chainer.links as L
import chainer.functions as F
from links import Elman, CRNN
import numpy as np
from chainer.functions.activation import tanh, relu, sigmoid

class Network(object):

    def add_monitor(self, monitor):

        # used to store computed states
        self.monitor.append(monitor)

    def reset_state(self):
        pass


class MLP(ChainList, Network):
    """
    Fully connected deep neural network consisting of a chain of layers (weight matrices)
    with a fixed number of nhidden units. Defaults to a standard MLP with one hidden layer.
    If n_hidden_layers=0 then we have a perceptron.

     """

    def __init__(self, n_input, n_output, n_hidden=10, n_hidden_layers=1, actfun=F.relu):
        """

        :param n_input: number of inputs
        :param n_output: number of outputs
        :param n_hidden: number of hidden units
        :param n_hidden_layers: number of hidden layers (1; standard MLP)
        :param actfun: used activation function (ReLU)
        """

        links = ChainList()
        if n_hidden_layers == 0:
            links.add_link(L.Linear(n_input, n_output))
        else:
            links.add_link(L.Linear(n_input, n_hidden))
            for i in range(n_hidden_layers - 1):
                links.add_link(L.Linear(n_hidden, n_hidden))
            links.add_link(L.Linear(n_hidden, n_output))

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_hidden_layers = n_hidden_layers
        self.actfun = actfun
        self.monitor = []

        super(MLP, self).__init__(links)

    def __call__(self, x, train=False):

        if self.n_hidden_layers == 0:

            y = self[0][0](x)

        else:

            if self.monitor:

                h = self.actfun(self[0][0](x))
                map(lambda x: x.set('hidden-1', h.data), self.monitor)
                for i in range(1,self.n_hidden_layers):
                    h = self.actfun(self[0][i](h))
                    map(lambda x: x.set('hidden-'+str(i+1), h.data), self.monitor)
                y = self[0][-1](h)
                map(lambda x: x.set('output', y.data), self.monitor)

            else:

                h = self.actfun(self[0][0](x))
                for i in range(1,self.n_hidden_layers):
                    h = self.actfun(self[0][i](h))
                y = self[0][-1](h)

        return y

#####
## Convolutional Neural Network

class ConvNet(Chain, Network):
    """
    Basic convolutional neural network
    """

    def __init__(self, n_input, n_output, n_hidden=10):
        """

        :param n_input: nchannels x height x width
        :param n_output: number of action outputs
        :param n_hidden: number of hidden units
        :param monitor: monitors internal states
        """

        k = 3 # kernel size
        s = 1 # stride
        p = 1 # padding
        n_linear = n_hidden/16 * np.prod(1 + (np.array(n_input[1:]) - k + 2*p)/s)
        super(ConvNet, self).__init__(
            l1=L.Convolution2D(n_input[0], n_hidden, k, s, p),
            l2=L.Convolution2D(n_hidden, n_hidden, k, s, p),

            l3=L.Linear(n_linear, n_output)
        )

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.monitor = []

    def __call__(self, x, train=False):
        """
        :param x: sensory input (ntrials x nchannels x ninput[0] x ninput[1])
        """


        h1=F.relu(self.l1(x))
        h2=F.max_pooling_2d(h1,2,stride=2)
        h3=F.relu(self.l2(h2))
        h4=F.max_pooling_2d(h3,2,stride=2)
        y = self.l3(h4)

        return y

#####
## Recurrent Neural Network

class RNN(ChainList, Network):
    """
    Recurrent neural network consisting of a chain of layers (weight matrices)
    with a fixed number of nhidden units
    nlayer determines number of layers. The last layer is always a linear layer. The other layers
    make use of an activation function actfun

    """

    def __init__(self, n_input, n_output, n_hidden=10, n_hidden_layers=1, link=L.LSTM):
        """

        :param n_input: number of inputs
        :param n_hidden: number of hidden units
        :param n_output: number of outputs
        :param n_hidden_layers: number of hidden layers
        :param link: used recurrent link (LSTM)

        """

        links = ChainList()
        if n_hidden_layers == 0:
            links.add_link(L.Linear(n_input, n_output))
        else:
            links.add_link(link(n_input, n_hidden))
            for i in range(n_hidden_layers - 1):
                links.add_link(link(n_hidden, n_hidden))
            links.add_link(L.Linear(n_hidden, n_output))

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_hidden_layers = n_hidden_layers
        self.monitor = []

        super(RNN, self).__init__(links)

    def __call__(self, x, train=False):

        if self.n_hidden_layers == 0:

            y = self[0][0](x)

        else:

            if self.monitor:

                h = self[0][0](x)
                map(lambda x: x.set('hidden-1', h.data), self.monitor)
                for i in range(1,self.n_hidden_layers):
                    h = self[0][i](h)
                    map(lambda x: x.set('hidden-'+str(i+1), h.data), self.monitor)
                y = self[0][-1](h)
                map(lambda x: x.set('output', y.data), self.monitor)

            else:

                h = self[0][0](x)
                for i in range(1, self.n_hidden_layers):
                    h = self[0][i](h)
                y = self[0][-1](h)

        return y

    def reset_state(self):
        for i in range(self.n_hidden_layers):
            self[0][i].reset_state()

class RNN2(Chain, Network):
    """
    Recurrent neural network consisting of a chain of layers (weight matrices)
    with a fixed number of nhidden units

    nlayer determines number of layers. The last layer is always a linear layer. The other layers
    make use of an activation function actfun

    """

    def __init__(self, n_input, n_output, n_hidden=10, n_hidden_layers=1, link=L.LSTM):
        """

        :param n_input: number of inputs
        :param n_hidden: number of hidden units
        :param n_output: number of outputs
        :param n_hidden_layers: number of hidden layers
        :param link: used recurrent link (LSTM)

        """



        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_hidden_layers = n_hidden_layers
        self.monitor = []

        super(RNN2, self).__init__(
            I1=link(n_input,n_hidden),
            I2=link(n_input,n_hidden,initOff=0),
            pi=L.Linear(n_hidden,n_output,initial_bias=np.asarray([0,0,0])),
            v = L.Linear(n_hidden, 1),
        )

    def __call__(self, x, train=False):

        if self.monitor:

            h1 = self.I1(x)
            h2 = self.I2(x)
            map(lambda x: x.set('hidden-1', h1.data), self.monitor)
            map(lambda x: x.set('hidden-2', h2.data), self.monitor)
            y1 = self.pi(h1)
            y2 = self.v(h2)
            y = F.concat((y1, y2))

            map(lambda x: x.set('output', y.data), self.monitor)

        else:

            h1 = self.I1(x)
            h2 = self.I2(x)
            y1 = self.pi(h1)
            y2 = self.v(h2)
            y = F.concat((y1, y2))

        return y1, y2

    def reset_state(self):
        self.I1.reset_state()
        self.I2.reset_state()

class CRNN2(Chain, Network):
    """
    Basic convolutional neural network
    """

    def __init__(self, n_input, n_output, n_hidden=10):
        """

        :param n_input: nchannels x height x width
        :param n_output: number of action outputs
        :param n_hidden: number of hidden units
        :param monitor: monitors internal states
        """

        k = 3  # kernel size
        s = 1  # stride
        p = 1  # padding
        n_linear = n_hidden/16 * np.prod(1 + (np.array(n_input[1:]) - k + 2 * p) / s)
        super(CRNN2, self).__init__(
            conv1=CRNN(n_input[0], n_hidden),
            conv2=CRNN(n_hidden, n_hidden),
            l2=L.Linear(n_linear, n_output)
        )

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.monitor = []

    def __call__(self, x, train=False):
        """
        :param x: sensory input (ntrials x nchannels x ninput[0] x ninput[1])
        """
        h1=self.conv1(x)
        h2=F.max_pooling_2d(h1, 2, stride=2)
        h3=self.conv2(h2)
        h4=F.max_pooling_2d(h3, 2, stride=2)
        y = self.l2(h4)

        return y

    def reset_state(self):
        self.conv1.reset_state()
        self.conv2.reset_state()

##### Convolutional RNN

class CRNN3(ChainList, Network):
    """
    Recurrent neural network consisting of a chain of layers (weight matrices)
    with a fixed number of nhidden units

    nlayer determines number of layers. The last layer is always a linear layer. The other layers
    make use of an activation function actfun

    """

    def __init__(self, n_input, n_output, n_hidden1=10, n_hidden2=10, n_hidden_layers=1, link=L.LSTM):
        """

        :param n_input: nchannels x height x width
        :param n_hidden: number of hidden units
        :param n_output: number of outputs
        :param n_hidden_layers: number of hidden layers
        :param link: used recurrent link (LSTM)

        """
        k = 3 # kernel size
        s = 1 # stride
        p = 1 # padding
        n_linear = n_hidden1 * np.prod(1 + (np.array(n_input[1:]) - k + 2*p)/s)
        links = ChainList()
        if n_hidden_layers == 0:
            links.add_link(L.Convolution2D(n_input[0], n_hidden1, k, s, p))
            links.add_link(L.Linear(n_linear, n_output))
        else:
            links.add_link(L.Convolution2D(n_input[0], n_hidden1, k, s, p))
            links.add_link(link(n_linear, n_hidden2))
            for i in range(n_hidden_layers - 1):
                links.add_link(link(n_hidden2, n_hidden2))
            links.add_link(L.Linear(n_hidden2, n_output))

        self.n_input = n_input
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2

        self.n_output = n_output
        self.n_hidden_layers = n_hidden_layers
        self.monitor = []

        super(CRNN3, self).__init__(links)

    def __call__(self, x, train=False):
        #:param x: sensory input (ntrials x nchannels x ninput[0] x ninput[1])


        if self.n_hidden_layers == 0:

            y = self[0][0](x)

        else:

            if self.monitor:

                h1 = self[0][0](x)
                h = self[0][1](h1)
                map(lambda x: x.set('hidden-1', h.data), self.monitor)
                for i in range(2, self.n_hidden_layers):
                    h = self[0][i](h)
                    map(lambda x: x.set('hidden-' + str(i + 1), h.data), self.monitor)
                y = self[0][-1](h)
                map(lambda x: x.set('output', y.data), self.monitor)

            else:

                h = self[0][0](x)
                for i in range(1, self.n_hidden_layers):
                    h = self[0][i](h)
                y = self[0][-1](h)

        return y

    def reset_state(self):
        for i in range(self.n_hidden_layers):
            self[0][i+1].reset_state()
#####
## Language model

class RNNForLM(Chain, Network):

    def __init__(self, n_vocab, n_hidden):

        super(RNNForLM, self).__init__(
            embed=L.EmbedID(n_vocab, n_hidden),
            l1=L.LSTM(n_hidden, n_hidden),
            l2=L.Linear(n_hidden, n_vocab),
        )

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

        self.monitor = []

    def __call__(self, x, train=False):

        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, train=train))
        y = self.l2(F.dropout(h1, train=train))

        return y

    def reset_state(self):
        self.l1.reset_state()

class RNNFBI(Chain, Network):
    """
    Basic convolutional neural network
    """

    def __init__(self, n_input, n_output, n_hidden=10):
        """

        :param n_input: nchannels x height x width
        :param n_output: number of action outputs
        :param n_hidden: number of hidden units
        :param monitor: monitors internal states
        """
        super(RNNFBI, self).__init__(
            l1=L.Linear(n_input, n_hidden),
            l2=L.Linear(n_hidden, n_output),
            l3=L.Linear(n_output,n_input)
        )

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.y=None
        self.monitor = []

    def __call__(self, x, train=False):
        """
        :param x: sensory input (ntrials x nchannels x ninput[0] x ninput[1])
        """
        if self.y is None:
            h1=self.l1(x)
        else:
            fb=relu.relu(self.l3(self.y))
            h1 = self.l1(fb)


        self.y = self.l2(relu.relu(h1))

        return self.y

    def reset_state(self):
        self.y=None

class ResRNN(Chain, Network):
    """
    Basic convolutional neural network
    """

    def __init__(self, n_input, n_output, n_hidden=10):
        """

        :param n_input: nchannels x height x width
        :param n_output: number of action outputs
        :param n_hidden: number of hidden units
        :param monitor: monitors internal states
        """

        k = 3  # kernel size
        s = 1  # stride
        p = 1  # padding
        n_linear = n_hidden * np.prod(1 + (np.array(n_input[1:]) - k + 2 * p) / s)
        super(ResRNN, self).__init__(
            conv1=CRNN(n_input[0], n_hidden),
            conv2=L.Convolution2D(n_hidden, n_hidden, k, s, p),
            conv3=L.Convolution2D(n_hidden, n_hidden, k, s, p),
            l2=L.Linear(n_linear, n_output)
        )

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.h2=None
        self.monitor = []

    def __call__(self, x, train=False):
        """
        :param x: sensory input (ntrials x nchannels x ninput[0] x ninput[1])
        """
        if self.h2 is None:
            h1 = self.conv1(x)
        else:
            h1 = self.conv3(self.h2)
        self.h2=self.conv2(h1)
        y = self.l2(self.h2)

        return y

    def reset_state(self):
        self.h2=None
        self.conv1.reset_state()
