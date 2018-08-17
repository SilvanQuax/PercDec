from chainer import Chain, Variable
import numpy as np
import chainer
from chainer.functions.activation import relu
from chainer import link
from chainer.links.connection import linear
from chainer import serializers
import chainer.functions as F
from chainer import cuda

#####
## Wrappers that compute the loss (negative objective function) and the final output for a neural network
#  The predictor is the neural network architecture which gives the predictions for which we can compute outputs and loss

class Model(Chain):

    def __init__(self, predictor):
        super(Model, self).__init__(predictor=predictor)

        self.monitor = []

    def add_monitor(self, monitor):

        # used to store computed states
        self.monitor.append(monitor)
        self.predictor.add_monitor(monitor)

    def load(self, fname):
        serializers.load_npz('{}'.format(fname), self)

    def save(self, fname):
        """ Save a model - boils down to saving network parameters plus any additional parameters to reconstruct the model

        :param fname:
        :return:
        """

        serializers.save_npz('{}'.format(fname), self)

    def __call__(self, data, train=False):
        """ Compute loss for minibatch of data

        :param data: list of minibatches (e.g. inputs and targets for supervised learning)
        :param train: call predictor in train or test mode
        :return: loss
        """

        raise NotImplementedError

    def predict(self, data, train=False):
        """
        Returns prediction, which can be different than raw output (e.g. for softmax function)

        :param data: minibatch or list of minibatches representing input to the model
        :param train: call predictor in train or test mode
        :return: prediction
        """

        raise NotImplementedError

    def reset(self):
        self.predictor.reset_state()

#####
## Supervised model
#

class SupervisedModel(Model):

    def __init__(self, predictor, loss_function=None, gpu=-1, output_function=lambda x:x):
        super(SupervisedModel, self).__init__(predictor=predictor)
        self.gpu=gpu
        self.loss_function = loss_function
        self.output_function = output_function
        global xp
        xp = np if gpu == -1 else cuda.cupy
        self.acc=0
    def __call__(self, data, train=False):

        x = data[0] if len(data)==3 else data[:-2] # inputs
        t = data[-2] # targets
        if self.gpu>0:
            x.to_gpu()
            t.to_gpu()
        # check for missing data
        # missing = [np.any(np.isnan(t[i].data)) or (t[i].data.dtype == 'int32' and np.any(t[i].data == -1)) for i in range(len(t.data))]

        if self.monitor:

            if isinstance(x, list):
                map(lambda v: v.set('input', np.hstack(map(lambda z: z.data, x))), self.monitor)
            else:
                map(lambda v: v.set('input', x.data), self.monitor)
            y = self.predictor(x, train=train)

            map(lambda x: x.set('prediction',xp.argmax(self.output_function(y).data,axis=1)), self.monitor)
            self.acc += xp.mean(xp.argmax(self.output_function(y).data, axis=1) == t.data)

            if self.gpu>0:
                map(lambda x: x.set('accuracy',cuda.to_cpu(xp.mean(xp.argmax(self.output_function(y).data,axis=1)==t.data))), self.monitor)
            else:
                map(lambda x: x.set('accuracy',xp.mean(xp.argmax(self.output_function(y).data,axis=1)==t.data)), self.monitor)

            loss = self.loss_function(y, t)

            map(lambda x: x.set('loss', loss.data), self.monitor)
            map(lambda x: x.set('target', t.data), self.monitor)

            return loss

        else:
            y = self.predictor(x, train=train)
            loss = self.loss_function(y, t)

            return loss

    def predict(self, data, train=False):

        if self.monitor:

            if isinstance(data, list):
                map(lambda v: v.set('input', np.hstack(map(lambda z: z.data, data))), self.monitor)
            else:
                map(lambda v: v.set('input', data.data), self.monitor)

            y = self.predictor(data, train=train)
            output = self.output_function(y).data

            map(lambda x: x.set('prediction', output), self.monitor)

            return output

        else:

            return self.output_function(self.predictor(data, train)).data


#####
## Classifier object

class Classifier(SupervisedModel):

    def __init__(self, predictor, gpu=-1):
        super(Classifier, self).__init__(predictor=predictor, loss_function=F.softmax_cross_entropy,
                                         output_function=F.softmax, gpu=gpu)

#####
## Regressor object

class Regressor(SupervisedModel):

    def __init__(self, predictor):
        super(Regressor, self).__init__(predictor=predictor, loss_function=F.mean_squared_error)


#####
## Reinforcement learning actor-critic model
#

class ActorCriticModel(Model):
    """
    An actor critic model computes the action, policy and value from a predictor
    """

    def __init__(self, predictor):
        super(ActorCriticModel, self).__init__(predictor=predictor)
        self.eps=0.5
        gpu=-1
        global xp
        xp = np if gpu == -1 else cuda.cupy
        self.acc = 0

    def __call__(self, x, train=False):
        """

        :param data: observation
        :param train: True or False
        :return: policy and value
        """

        # separate observation from reward
        #x = data[0] if len(data) == 3 else data[:-2]  # inputs (rest is reward and terminal state)

        if self.monitor:
            # if isinstance(x, list):
            #     map(lambda v: v.set('input', np.hstack(map(lambda z: z.data, x))), self.monitor)
            # else:
                map(lambda v: v.set('input', x.data), self.monitor)

        # linear outputs reflecting the log action probabilities and the value
        out1, out2 = self.predictor(x, train)
        policy=out1[:,:]
        value=out2[:,-1]
        #
        # policy = out[:,:-1]
        #
        # value = out[:,-1]

        # # handle case where we have only one element per batch
        # if value.ndim == 1:
        #     value = F.expand_dims(value, axis=1)

        action = self.get_action(policy,train)

        if self.monitor:
            map(lambda x: x.set('action', action), self.monitor)

        return action, policy, value

    def predict(self, data, train=False):

        out = self.predictor(data, train)

        policy = out[:, :-1]

        return self.get_action(policy)

    def get_action(self,policy,train):

        # generate action according to policy
        p = F.softmax(policy).data

        # normalize p in case tiny floating precision problems occur
        row_sums = p.sum(axis=1)
        p /= row_sums[:, np.newaxis]

        if self.monitor:
            map(lambda x: x.set('p1', p[0,0]), self.monitor)
            map(lambda x: x.set('p2', p[0,1]), self.monitor)
            map(lambda x: x.set('p3', p[0,-1]), self.monitor)
        # discrete representation
        n_out = self.predictor.n_output
        self.eps=self.eps*0.9995
        if np.random.rand()<0 and train:
            action = np.array([np.random.choice(n_out, None, True, [0.05,0.05,0.9])])
        else:
            action = xp.array([xp.random.choice(n_out, 1, True, p[i]) for i in range(p.shape[0])], dtype='int32')
            action=action[0]
            ###test###
            #action = xp.array([xp.argmax(p)], dtype='int32')
        return action



class Model2(Chain):

    def __init__(self, predictor1, predictor2):
        super(Model2, self).__init__(predictor1=predictor1, predictor2=predictor2)

        self.monitor = []

    def add_monitor(self, monitor):

        # used to store computed states
        self.monitor.append(monitor)
        self.predictor1.add_monitor(monitor)
        self.predictor2.add_monitor(monitor)

    def load(self, fname):
        serializers.load_npz('{}'.format(fname), self)

    def save(self, fname):
        """ Save a model - boils down to saving network parameters plus any additional parameters to reconstruct the model

        :param fname:
        :return:
        """

        serializers.save_npz('{}'.format(fname), self)

    def __call__(self, data, train=False):
        """ Compute loss for minibatch of data

        :param data: list of minibatches (e.g. inputs and targets for supervised learning)
        :param train: call predictor in train or test mode
        :return: loss
        """

        raise NotImplementedError

    def predict(self, data, train=False):
        """
        Returns prediction, which can be different than raw output (e.g. for softmax function)

        :param data: minibatch or list of minibatches representing input to the model
        :param train: call predictor in train or test mode
        :return: prediction
        """

        raise NotImplementedError

    def reset(self):
        self.predictor1.reset_state()
        self.predictor2.reset_state()

class ActorCriticModel2(Model2):
    """
    An actor critic model with separate network for value and policy
    """

    def __init__(self, predictor1, predictor2):
        super(ActorCriticModel2, self).__init__(predictor1=predictor1, predictor2=predictor2)
        #self.predictor2=predictor2
        self.eps=0.5

    def __call__(self, data, train=False):
        """

        :param data: observation
        :param train: True or False
        :return: policy and value
        """

        # separate observation from reward
        x = data[0] if len(data) == 3 else data[:-2]  # inputs (rest is reward and terminal state)

        if self.monitor:
            if isinstance(x, list):
                map(lambda v: v.set('input', np.hstack(map(lambda z: z.data, x))), self.monitor)
            else:
                map(lambda v: v.set('input', x.data), self.monitor)

        # linear outputs reflecting the log action probabilities and the value
        policy = self.predictor1(x, train)

        value = self.predictor2(x, train)

        # # handle case where we have only one element per batch
        # if value.ndim == 1:
        #     value = F.expand_dims(value, axis=1)

        action = self.get_action(policy,train)

        if self.monitor:
            map(lambda x: x.set('action', action), self.monitor)
            # if train == False:
            #     map(lambda x: x.set('h', self.predictor1.h[0][0]), self.monitor)

        return action, policy, value[:,0]

    def predict(self, data, train=False):

        policy = self.predictor1(data, train)

        return self.get_action(policy)

    def get_action(self,policy,train):

        # generate action according to policy
        p = F.softmax(policy).data

        # normalize p in case tiny floating precision problems occur
        row_sums = p.sum(axis=1)
        p /= row_sums[:, np.newaxis]

        if self.monitor:
            map(lambda x: x.set('p1', p[0,0]), self.monitor)
            map(lambda x: x.set('p2', p[0,1]), self.monitor)
            map(lambda x: x.set('p3', p[0,-1]), self.monitor)
        # discrete representation
        n_out = self.predictor1.n_output
        self.eps=self.eps*0.99995
        if np.random.rand()<0 and train:
            action = np.array([np.random.choice(n_out, None, True, [0.05,0.05,0.9])])
        else:
            action = np.array([np.random.choice(n_out, None, True, p[i]) for i in range(p.shape[0])], dtype='int32')

        return action


