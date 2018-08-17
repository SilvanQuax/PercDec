from base import Iterator
import numpy as np
import copy
import chainer.cuda
import time


###
# Simple MDP task

class Foo(Iterator):
    """
    Very simple environment for testing fully observed models. The actor gets a reward when it correctly decides
    on the ground truth. Ground truth 0/1 determines probabilistically the number of 0s or 1s as observations

    """

    def __init__(self, n=2, p=0.8, batch_size=1, n_batches=np.inf):
        """

        :param n: number of inputs
        :param p: probability of emitting the right sensation at the input
        :param batch_size: one run by default
        :param n_batches: infinite number of time steps by default
        """

        super(Foo, self).__init__(batch_size=batch_size, n_batches=n_batches)

        self.n_input = n
        self.p = p

        self.n_action = 1  # number of action variables
        self.n_output = 2  # number of output variables (actions) for the agent (discrete case)
        self.n_states = 1  # number of state variables

    def __iter__(self):

        self.idx = -1
        self.state = self.get_state()
        self.obs = self.get_observation()
        self.reward = None
        self.terminal = None

        return self

    def next(self):

        if self.idx == self.n_batches - 1:
            raise StopIteration

        self.idx += 1

        # return new observation and reward associated with agent's choice, we are always in a terminal state
        return self.obs, self.reward, np.ones(self.batch_size, dtype=np.bool)

    def process(self, agent):
        """ Process agent action, compute reward and generate new state and observation

        :param agent:
        :return:
        """

        if self.monitor:
            map(lambda x: x.set('accuracy', 1.0 * np.sum(agent.action == self.state) / agent.action.size), self.monitor)

        self.reward = (2 * (agent.action == self.state) - 1).astype('float32')

        # this task always produces a new observation after each decision
        self.state = self.get_state()

        self.obs = self.get_observation()

    def get_state(self):
        """

        :return: new state
        """

        return np.random.choice(2, self.batch_size, True, [0.5, 0.5]).astype('int32')

    def get_observation(self):
        """

        :return: observation given the state
        """

        # produce a new observation at each step
        obs = np.zeros([self.batch_size, self.n_input]).astype('float32')
        for i in range(self.batch_size):

            if self.state[i] == 0:
                obs[i] = np.random.choice(2, [1, self.n_input], True, np.array([self.p, 1 - self.p]))
            else:
                obs[i] = np.random.choice(2, [1, self.n_input], True, np.array([1 - self.p, self.p]))

        return obs

class DataTaskMnist(Iterator):

    def __init__(self, data, data2=None, batch_size=1, n_batches=np.inf, coh_all=[0], rewards=[-1, 10, -10],
                 noise_method=0, fix_length=10, trial_length=100):

        batch_size = batch_size or len(data)

        super(DataTaskMnist, self).__init__(batch_size=batch_size, n_batches=n_batches)

        self.data = data
        self.data2 = data
        self.fix_length = fix_length
        self.trial_length = trial_length
        self.n_samples = len(data)
        self.n_input = data[0][0].size

        # number of actions. Last action is the decision to accumulate more information
        self.n_output = np.unique(map(lambda x: x[1], data)).size + 1

        # flags noise level
        self.coh_all = coh_all
        self.noise_method = noise_method

        # rewards/costs for asking for a new observation, deciding on the right category, deciding on the wrong category
        self.rewards = rewards
        self.cum_reward = np.zeros(1)


    def __iter__(self):

        self.idx = -1
        self.idxt = -1
        self.idx10 = []

        # generate another random batch in each epoch
        _order = np.random.permutation(self.n_samples)[:1]

        # keep track of true class
        self.state = self.data[_order][1]

        # generate data
        self.obs = self.data[_order][0]
        self.coh = self.coh_all[np.random.randint(0, len(self.coh_all))]

        self.reward = None

        self.terminal = np.zeros(self.batch_size, dtype=np.bool)

        self.statefix = 1

        self.rand = np.random.randint(-5, high=5)
        self.mean_image = np.zeros((2, 51, 784))
        self.mean_image_count = np.zeros((2, 51))
        self.tmi = np.zeros((51, 784))
        self.tmic = np.zeros(51)

        return self

    def next(self):

        if self.idx == self.n_batches - 1:
            raise StopIteration

        if self.terminal == 1:
            self.idxt = -1
            self.statefix = 1
            self.rand = np.random.randint(-5, high=5)

            self.tmi = np.zeros((51, 784))
            self.tmic = np.zeros(51)

        self.idx += 1
        self.idxt += 1

        # set fixation to 0 when fixation period over
        if self.idxt == self.fix_length:
            self.statefix = 0

        if self.monitor:
            map(lambda x: x.set('state', copy.copy(self.state)), self.monitor)
            map(lambda x: x.set('coh', copy.copy(self.coh)), self.monitor)
            map(lambda x: x.set('step', copy.copy(self.idxt)), self.monitor)

        if self.statefix == 1:
            noise_obs = np.zeros((1, len(self.obs[0]))).astype('float32')

        elif self.noise_method == 1:
            noise_obs = self.add_noise(np.copy(self.obs))
        elif self.noise_method == 2:
            noise_obs = self.add_noise2(np.copy(self.obs))

        if self.monitor:
            map(lambda x: x.set('nobs', copy.copy(noise_obs)), self.monitor)

        # save stimuli for PK calculation
        # self.mean_image[self.state,self.idxt,:]+=noise_obs
        self.tmi[self.idxt, :] = noise_obs
        self.tmic[self.idxt] = 1

        return noise_obs, self.reward, self.terminal

    def add_noise(self, data):

        d_shape = data.shape
        d_size = data.size

        # create noise component
        noise = np.zeros(d_size)
        n = int(np.ceil(self.coh * d_size))
        noise[np.random.permutation(d_size)[:n]] = np.random.rand(n)
        noise = noise.reshape(d_shape)

        data[noise != 0] = noise[noise != 0]

        return data

    def add_noise2(self, data):

        d_shape = data.shape

        data = data + self.noise * np.random.randn(d_shape[0], d_shape[1])

        return data.astype('float32')

    def process(self, agent):
        """ Process agent action, compute reward and generate new state and observation

        :param agent:
        :return:
        """
        if agent.action.squeeze() != self.n_output - 1 and self.idxt == 20:
            self.mean_image_count[agent.action.squeeze(), :] += self.tmic
            self.mean_image[agent.action.squeeze(), :, :] += self.tmi
            self.idx10.append(self.idx)

        self.reward = np.zeros(len(agent.action), dtype=np.float32)

        # When fixation on you get negative reward for choosing
        if self.statefix == 1:
            if agent.action.squeeze() == self.n_output - 1:
                self.reward = self.rewards[0]
            else:
                self.reward = 0

        # When fixation off you get reward based on good classification
        if self.statefix == 0:
            if agent.action.squeeze() == self.n_output - 1:
                self.reward = self.rewards[0]
            elif agent.action.squeeze() == self.state:
                self.reward = self.rewards[1]
            elif agent.action.squeeze() != self.state:
                self.reward = self.rewards[2]

        # Determine whether in terminal state
        if agent.action.squeeze() == self.n_output - 1:
            self.terminal = 0
        else:
            self.terminal = 1

            _order = np.random.permutation(self.n_samples)[:1]
            if self.idx < 100000:
                self.state = self.data[_order][1]
                self.obs = self.data[_order][0]
            else:
                self.state = self.data2[_order][1]
                self.obs = self.data2[_order][0]
            self.coh = self.coh_all[np.random.randint(0, len(self.coh_all))]


        # stop if trial longer then certain amount of steps
        if self.idxt == self.trial_length:
            self.reward = 0
            self.terminal = 1
            # print(self.idxt)

            _order = np.random.permutation(self.n_samples)[:1]
            self.state = self.data[_order][1]
            self.obs = self.data[_order][0]
            self.coh = self.coh_all[np.random.randint(0, len(self.coh_all))]

        self.cum_reward += self.reward

        # compute accuracy on those trials for which a decision is being made
        if self.monitor:
            if self.terminal == 1 and self.statefix == 0 and self.idxt < self.trial_length:
                map(lambda x: x.set('acc_trial', int(agent.monitor[2]['action'][-1] == self.monitor[0]['state'][-1])),
                    self.monitor)
            elif self.terminal == 1:
                map(lambda x: x.set('acc_trial', None), self.monitor)

        if self.monitor and self.idx % 1000 == 0:

            if len(self.monitor) > 1:

                if self.idx > 1000 and (1000 - (np.sum(agent.monitor[2]['action'][-1000:] == (self.n_output - 1)))) > 0:
                    ind_tr = np.where(agent.monitor[2]['action'][-1000:] != (self.n_output - 1))
                    step_seq = self.monitor[0]['step'][-1000:]
                    non_resp = np.sum(step_seq[ind_tr] < self.fix_length) + np.sum(
                        self.monitor[0]['step'][-1000:] == self.trial_length)

                    map(lambda x: x.set('accuracy', 1.0 * np.sum(
                        agent.monitor[2]['action'][-1000:] == self.monitor[0]['state'][-1000:]) / (
                                            np.sum(agent.monitor[2]['action'][-1000:] != (self.n_output - 1)))),
                        self.monitor)
                    map(lambda x: x.set('resp_ratio',
                                        1 - (1.0 * non_resp / np.sum(np.diff(self.monitor[0]['step'][-1000:]) < 1))),
                        self.monitor)
                    map(lambda x: x.set('reward_rate', self.cum_reward / np.sum(step_seq == 0)), self.monitor)
                else:
                    map(lambda x: x.set('accuracy', 0), self.monitor)
                    map(lambda x: x.set('resp_ratio', 0), self.monitor)
                    map(lambda x: x.set('reward_rate', 0), self.monitor)
                self.cum_reward = np.zeros(1)


class DynamicSupervised(Iterator):

    def __init__(self, data, noise_all=[0], batch_size=1, n_batches=np.inf, noise_method=0, cnoise=False, gpu=-1):

        super(DynamicSupervised, self).__init__(batch_size=batch_size, n_batches=n_batches)
        if gpu <= 0:
            self.xp = np
        else:
            self.xp = chainer.cuda.cupy

        batch_size = batch_size or len(data)
        self.data = data
        self.noise_all = noise_all
        self.n_samples = len(data)
        self.n_input = data[0][0].size
        self.cnoise = cnoise
        # number of actions. Last action is the decision to accumulate more information
        self.n_output = self.xp.unique(map(lambda x: x[1], data)).size
        self.gpu = gpu
        # flags noise level
        self.noise_method = noise_method

    def __iter__(self):

        self.idx = -1

        # generate another random batch in each epoch
        _order = self.xp.random.permutation(self.n_samples)[:self.batch_size]

        # keep track of true class
        self.state = self.data[_order][1]

        # generate data
        if self.gpu <= 0:
            self.obs = self.data[_order][0]
        else:
            self.obs = chainer.cuda.to_gpu(self.data[_order][0])

        self.noise = self.noise_all[self.xp.random.randint(0, len(self.noise_all), size=1)]
        self.terminal = 1

        return self

    def next(self):

        if self.idx == self.n_batches - 1:
            raise StopIteration

        self.idx += 1

        if self.terminal == 1:
            self.pres_obs = np.zeros(self.obs.shape).astype('float32')
            if self.cnoise == True:
                if self.noise_method == 1:
                    self.noise_obs = self.add_noise(self.xp.copy(self.obs))
                elif self.noise_method == 2:
                    self.noise_obs = self.add_noise2(self.xp.copy(self.obs))
                else:
                    self.noise_obs = self.obs
        else:
            if self.cnoise == True:
                self.pres_obs = self.noise_obs
            # else:
            #     self.noise_obs=self.xp.zeros(self.obs.shape).astype('float32')
            else:
                if self.noise_method == 1:
                    self.noise_obs = self.add_noise(self.xp.copy(self.obs))
                elif self.noise_method == 2:
                    self.noise_obs = self.add_noise2(self.xp.copy(self.obs))
                self.pres_obs = self.noise_obs

        if self.monitor:
            map(lambda x: x.set('state', copy.copy(self.state)), self.monitor)
            map(lambda x: x.set('noise', copy.copy(self.noise)), self.monitor)

        return self.pres_obs, self.state, self.terminal

    def add_noise(self, data):

        d_shape = data.shape
        d_size = data.size

        noise = self.xp.zeros(d_size)
        n = int(self.xp.ceil(self.noise * d_size))
        noise[self.xp.random.permutation(d_size)[:n]] = self.xp.random.rand(n)
        noise = noise.reshape(d_shape)
        data[noise != 0] = noise[noise != 0]

        return data

    def add_noise2(self, data):

        d_shape = data.shape

        data = data + (self.noise * self.xp.random.randn(d_shape[0], d_shape[1])).astype('float32')

        return data

    def process(self, agent):
        """ Process agent action, compute reward and generate new state and observation

        :param agent:
        :return:
        """

        # compute which ones are terminal states
        if (self.idx + 1) % agent.cutoff == 0:
            self.terminal = 1
            _order = self.xp.random.permutation(self.n_samples)[:self.batch_size]
            self.state = self.data[_order][1]
            if self.gpu <= 0:
                self.obs = self.data[_order][0]
            else:
                self.obs = chainer.cuda.to_gpu(self.data[_order][0])
            self.noise = self.noise_all[self.xp.random.randint(0, len(self.noise_all), size=1)]

        else:
            self.terminal = 0
