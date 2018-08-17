from __future__ import division

from chainer import Variable
from agent.base import Agent
import chainer

class StatelessAgent(Agent):

    def run(self, data, train=True, idx=None, final=False):

        loss = self.model(map(lambda x: Variable(self.xp.asarray(x)), data), train=train)

        # normalize by number of datapoints in minibatch
        _loss = float(loss.data/data[0].shape[0])

        if train:

            self.optimizer.zero_grads()
            loss.backward()
            self.optimizer.update()

        return _loss

class StatefulAgent(Agent):

    def __init__(self, model, optimizer=None, gpu=-1, cutoff=None, last=False):

        super(StatefulAgent, self).__init__(model, optimizer=optimizer, gpu=gpu, last=last, cutoff=cutoff)

        # cutoff for BPTT
        self.cutoff = cutoff

        # whether to update from loss in last step only
        self.last = last

        # keep track of loss for truncated BPTT
        self.loss = Variable(self.xp.zeros((), 'float32'))

    def run(self, data, train=True, idx=None, final=False):


        if (idx) % self.cutoff == 0:
            self.reset()

        loss = self.model(map(lambda x: Variable(self.xp.asarray(x)), data), train=True)

        if self.last:  # used in case we propagate back at end of trials only
            if ((idx + 1) % self.cutoff) == 0:
                self.loss = loss
            else:
                loss = Variable(self.xp.zeros((), 'float32'))
        else:
            self.loss += loss

        # normalize by number of datapoints in minibatch
        _loss = float(loss.data)

        # backpropagate if we reach the cutoff for truncated backprop or if we processed the last batch
        if train and ((self.cutoff and ((idx+1) % self.cutoff) == 0) or final):
            self.optimizer.zero_grads()
            self.loss.backward()
            self.loss.unchain_backward()
            self.optimizer.update()
            self.loss = Variable(self.xp.zeros((), 'float32'))

        if not train:
            self.loss.unchain_backward()

        return _loss