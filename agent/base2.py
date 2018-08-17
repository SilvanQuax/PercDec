from __future__ import division

import numpy as np
from chainer import cuda
import copy

class Agent(object):

    def __init__(self, model, optimizer=None, gpu=-1, last = False, cutoff=None):
        """

        :param model:
        :param optimizer:
        :param gpu:
        """

        self.model = model

        if not optimizer is None:

            optimizer.setup(model)

            # facilitate setting of hooks
            # optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

        self.optimizer = optimizer

        self.xp = np if gpu == -1 else cuda.cupy

        self.reset()

        self.last = last

        self.cutoff = cutoff

        self.monitor = []

    def __deepcopy__(self, memodict={}):
        # used to create copied agents for validation purposes
        c_agent = self.__class__(copy.deepcopy(self.model), copy.deepcopy(self.optimizer))
        c_agent.xp = self.xp
        c_agent.last = self.last
        c_agent.cutoff = self.cutoff

        return c_agent

    def add_monitor(self, monitor):

        # used to store computed states
        self.monitor.append(monitor)
        self.model.add_monitor(monitor)

    def reset(self):
        self.model.reset()

    def run(self, batch, train=True, idx=None, final=False):
        """ Process one minibatch

        :param batch: minibatch
        :param train: run agent in train or in test mode
        :param idx: index of current batch
        :param final: flags if we are in the final batch
        :return: loss for this minibatch averaged over number of datapoints
        """

        raise NotImplementedError
