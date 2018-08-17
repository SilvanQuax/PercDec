from __future__ import division

import os
import copy
import numpy as np
import tqdm
from brain.monitor import Oscilloscope
from chainer import cuda
#####
## An iterator generates batches

class Iterator(object):

    def __init__(self, batch_size=None, n_batches=None):

        self.batch_size = batch_size
        self.n_batches = n_batches

        self.monitor = []

    def __iter__(self):

        self.idx = -1

        return self

    def next(self):
        """

        :return: a list of numpy arrays where the first dimension is the minibatch size
        """

        raise NotImplementedError

    def is_final(self):
        """

        :return: boolean if final batch is reached
        """
        return (self.idx==self.n_batches-1)

    def process(self, agent):
        """
        Processing the possible actions of an agent
        By default this has no effect

        :param agent:
        :return:
        """

        raise NotImplementedError

    def add_monitor(self, monitor):
        # used to store computed states
        self.monitor.append(monitor)

    def render(self, agent):
        """TO DO: Rendering function to track input-output over time

        :param agent:
        :return:
        """
        pass

#####
## World class

class World(object):

    def __init__(self, agents, out='result', labels=None):
        """ A world is inhabited by one or more agents

        :param agents:
        :param out: output folder
        :param labels: labels for line plots
        """

        if not isinstance(agents, list):
            self.agents = [agents]
        else:
            self.agents = agents

        self.out = out
        if not self.out is None:
            try:
                os.makedirs(self.out)
            except OSError:
                pass

        self.n_agents = len(self.agents)


        # optional labels for plotting
        # Note that validate follows the order ['training-0', 'validation-0', 'training-1', 'validation-1']
        self.labels = labels

    def train(self, data_iter, n_epochs=1, plot=0, snapshot=0, monitor=0):
        self.run(data_iter, val_iter=None, train=True, n_epochs=n_epochs, plot=plot, snapshot=snapshot, monitor=monitor)

    def test(self, data_iter, n_epochs=1, plot=0, snapshot=0, monitor=0):
        self.run(data_iter, val_iter=None, train=False, n_epochs=n_epochs, plot=plot, snapshot=snapshot, monitor=monitor)

    def validate(self, data_iter, validation, n_epochs=1, plot=0, snapshot=0, monitor=0):
        self.run(data_iter, val_iter=validation, train=True, n_epochs=n_epochs, plot=plot, snapshot=snapshot, monitor=monitor)

    def run(self, data_iter, val_iter=None, train=False, n_epochs=1, plot=0, snapshot=0, monitor=0):
        """ Used to train, test and validate a model. Generalizes to the use of multiple agents

        :param data_iter: environment to train on
        :param val_iter: environment to validate on
        :param train: run in train or test mode
        :param n_epochs: number of epochs to run an environment
        :param plot: plot change in loss - -1 : per epoch; > 0 : after this many iterations
        :param snapshot: save snapshot - -1 : per epoch; > 0 : after this many iterations
        :param monitor: execute monitor.run() - -1 : per epoch; > 0 : after this many iterations
        :return:
        """

        # initialize plotting of loss
        if plot:
            labels = self.get_labels(train, val_iter)
            loss_monitor = Oscilloscope(ylabel='loss', names=labels)

        # initialize iterator
        d_it = iter(data_iter)

        # copy agents for purpose of validation
        if val_iter:
            val_agents = map(lambda x: copy.deepcopy(x), self.agents)
            val_losses = np.zeros(self.n_agents)
            v_it = iter(val_iter)

        # initialization for validation
        min_loss = [None] * self.n_agents
        optimal_model = [None] * self.n_agents

        # maximal number of iterations is epochs * nr of batches (possibly infinite in case of task)
        # In case of infinity, we cap iterations at maximal integer
        max_iter = int(np.min([n_epochs * data_iter.n_batches, np.iinfo(np.int32).max]))

        # initialize losses
        losses = np.zeros(self.n_agents)


        map(lambda x: x.reset(), self.agents)

        if val_iter:
            map(lambda x: x.reset(), val_agents)
        # iterate over indices
        for epoch in xrange(0,n_epochs):

            self.train()


                # run validation
            if not val_iter is None:
                for _iter in tqdm.tqdm(xrange(0, max_iter)):

                    # copy parameters of trained model
                    for i in range(self.n_agents):
                        val_agents[i].model.copyparams(self.agents[i].model)

                    # run on data point
                    try:
                        val_data = v_it.next()
                    except StopIteration:
                        v_it = iter(val_iter)
                        val_data = v_it.next()

                    # compute validation loss
                    val_losses += map(lambda x: x.run(val_data, idx=_iter,train=False), val_agents)

                    map(lambda x: val_iter.process(x), val_agents)

                # store best models in case we are validating
                for i in range(self.n_agents):

                    if min_loss[i] is None:
                        optimal_model[i] = copy.deepcopy(val_agents[i].model)
                        min_loss[i] = val_losses[i]
                    else:
                        if val_losses[i] < min_loss[i]:
                            optimal_model[i] = copy.deepcopy(val_agents[i].model)
                            min_loss[i] = val_losses[i]
                    # is loss minimum best model?
                    optimal_model[i] = copy.deepcopy(val_agents[i].model)

            # plot loss - handled by a class-specific monitor
            if plot > 0 and _iter > 0 and _iter % plot == 0:
                for i in range(len(self.agents)):
                    loss_monitor.set(labels[2*i], losses[i] / (plot))
                    if not val_iter is None:
                        loss_monitor.set(labels[2*i+1], val_losses[i] / plot)
                losses = np.zeros(self.n_agents)
                val_losses = np.zeros(self.n_agents)
                loss_monitor.run()

            # store model
            self.save_snapshot(_iter, snapshot)

            # if monitor is defined then run optional monitoring function
            self.run_monitors(_iter, monitor)

        # each agent is assigned the best 'brain' according to validation loss
        if not val_iter is None:
            for i in range(self.n_agents):
                self.agents[i].model = optimal_model[i]
                self.agents[i].optimizer.target = optimal_model[i]

        if plot:
            loss_monitor.save(os.path.join(self.out, 'loss'))

    def train(self, max_iter):

        for _iter in tqdm.tqdm(xrange(0, max_iter)):

            # reset agents at start of each epoch


            try:
                data = d_it.next()
            except StopIteration:
                d_it = iter(data_iter)
                data = d_it.next()

            losses += map(lambda x: x.run(data, train=train, idx=_iter, final=data_iter.is_final()),
                          self.agents)

            # the iterator can process actions of the agents that change the state of the iterator
            # used in case of processing of tasks by RL agents
            map(lambda x: data_iter.process(x), self.agents)

    def save_snapshot(self, idx, snapshot):
        if snapshot > 0 and idx > 0 and idx % snapshot == 0:
            for i in range(self.n_agents):
                self.agents[i].model.save(os.path.join(self.out, 'agent-{0:04d}-snapshot-{1:04d}'.format(i, idx)))

    def run_monitors(self, idx, monitor):
        if monitor > 0 and idx > 0 and idx % monitor == 0:
            map(lambda x: map(lambda z: z.run(), x.monitor) if x.monitor else None, self.agents)

    def get_labels(self, train, validation):

        if not self.labels is None:
            labels = self.labels
        else:
            if len(self.agents)==1:
                if train:
                    labels = ['training']
                else:
                    labels = ['testing']
                if not validation is None:
                    labels += ['validation']
            else:
                labels = []
                for i in range(len(self.agents)):
                    if train:
                        labels += ['training-' + str(i)]
                    else:
                        labels += ['testing-' + str(i)]
                    if not validation is None:
                        labels += ['validation-' + str(i)]

        return labels