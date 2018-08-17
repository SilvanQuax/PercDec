from __future__ import division

from chainer import Variable
from agent.base import Agent
from brain.monitor import Monitor
import numpy as np
import chainer.functions as F
import copy

#####
## Actor-critic agent
#

class ActorCriticAgent(Agent):
    """

    Implements advantage actor critic and REINFORCE (which does not use a value baseline)

    Note that REINFORCE is a policy gradient method which does not use a critic.
    Instead the return is computed as a running estimate

    https://webdocs.cs.ualberta.ca/%7Esutton/book/bookdraft2016sep.pdf
    https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient
    http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/
    http://www.1-4-5.net/~dmm/ml/log_derivative_trick.pdf
    """

    def __init__(self, model, optimizer=None, gpu=-1, cutoff=None, gamma=0.975, beta=1e-2, aac=True):
        """

        :param model:
        :param optimizer:
        :param gpu:
        :param cutoff:
        :param gamma:
        :param beta:
        :param aac: Advantage actor-critic (True) or REINFORCE (false)
        """

        # cutoff for truncated BPTT
        self.cutoff = cutoff

        # discounting factor
        self.gamma = gamma

        # contribution of entropy term
        self.beta = beta

        # monitor score, entropy and reward
        self.buffer = Monitor()

        # keep track of cumulative reward
        self.cum_reward = 0

        # AAC mode
        self.aac = aac

        self.loss = 0

        self.trial = 0

        super(ActorCriticAgent, self).__init__(model, optimizer=optimizer, gpu=gpu)

    def run(self, data, train=True, idx=None, final=False):
        """

        :param data: a new observation and the reward associated with the previous observation and action
        :param train:
        :param idx:
        :param final:
        :return:
        """

        # This code only optimally processes single runs of a task
        # Reason is that we cannot process easily in parallel when using terminal states
        # This requires resetting of internal states per batch index
        # Note that we might gain a speed-up by completely ignoring terminal states and
        # have the agent use its observation to identify whether it started a new trial
        # problem is that this does not allow convergence in e.g. probabilistic categorization task
        # On the other hand, in lifelong learning settings we never have terminal states. This requires more thought
        # assert(data.batch_size==1)

        # get reward associated with taking the previous action in the previous state
        reward = data[-2]
        if not reward is None:
            self.buffer.set('reward', reward)
            self.cum_reward += reward

        # get terminal state
        terminal = data[-1]


        self.beta=self.beta*0.99998
        # backpropagate if we reach the cutoff for truncated backprop or if we processed the last batch
        if train and idx > 0 and ((self.cutoff and (idx % self.cutoff) == 0) or final or (terminal)):

            # store cumulative reward
            if self.monitor:
                map(lambda x: x.set('cumulative reward', np.mean(self.cum_reward)), self.monitor)

            # trial is finished when terminal state is encountered
            self.trial += 1
            trial_length=len(self.buffer.dict['reward'])
            # return value associated with last state
            #if terminal or not self.aac:
            R = Variable(self.xp.asarray([0]).astype('float32'))
            # else:
            #     R = value
            pi_loss = v_loss = 0
            T=len(self.buffer.dict['reward'])

            for i in range(len(self.buffer.dict['reward'])-1,-1,-1):

                R = self.buffer.dict['reward'].pop() + self.gamma * R

                if self.aac:
                    advantage = R - self.buffer.dict['value'].pop()

                    _ss = F.squeeze(self.buffer.dict['score'].pop(), axis=1) * advantage.data #correct?
                else:
                    _ss = F.squeeze(self.buffer.dict['score'].pop(),axis=1) * R

                if _ss.size > 1:
                    _ss = F.sum(_ss, axis=0)
                pi_loss -= F.squeeze(_ss)

                pi_loss -= self.beta * self.buffer.dict['entropy'].pop()

                if self.aac:
                    v_loss += F.sum(advantage ** 2)

            # Compute total loss; 0.5 supposedly used by Mnih et al
            if self.aac:
                v_loss = F.reshape(v_loss, pi_loss.data.shape)
                self.loss += (pi_loss  + 0.5 * v_loss)
            else:
                self.loss += pi_loss


            if self.trial == 30:
                self.loss=self.loss
                _loss = pi_loss.data
                self.optimizer.zero_grads()
                self.loss.backward()
                self.optimizer.update()
                self.loss.unchain_backward()
                self.trial = 0
                self.loss = 0
            else:

                _loss = 0

            # store return
            if self.monitor:
                map(lambda x: x.set('return', np.mean(R.data)), self.monitor)


        else:
            _loss = 0

        # reset hidden state when trial is finished

        if terminal:
            self.reset()


        # compute policy and take new action based on observations
        input = data[0] if len(data) == 3 else data[:-2]

        self.action, policy, value = self.model(Variable(self.xp.asarray(input)), train=train)

        # recompute score function: grad_theta log pi_theta (s_t, a_t) * v_t
        self.buffer.set('score', self.score_function(self.action, policy))

        # compute entropy
        self.buffer.set('entropy', F.sum(self.entropy(policy)))

        # add value
        if self.aac:
            self.buffer.set('value', value)

        return _loss

    def score_function(self, action, pi):

        logp = F.log_softmax(pi)

        score = F.select_item(logp, Variable(action))
        if score.data==0:
            print(score.data)
            print(pi.data)
        # handle case where we have only one element per batch
        if score.ndim == 1:
            score = F.expand_dims(score, axis=1)

        return score

    def entropy(self, pi):
        #A=Variable(np.asarray([0,0,0,0,0,0,0,0,0,0,3]).reshape((1,11)).astype('float32'))
        #pi=pi-A
        # pi.data[-1]=0.1*pi.data[-1]
        p = F.softmax(pi) #*Variable(np.array([[1,1,0.5]].astype('float32'))
        logp = F.log_softmax(pi)

        return - F.sum(p * logp, axis=1)