# test object categorization under noise as an RL task
import sys

sys.path.append('/home/squax/GitHub/CMS/')
from agent.reinforcement import *
from brain.monitor import *
from brain.models import ActorCriticModel
from brain.networks import *
from world.base2 import World
from world.data import *
from world.tasks import *
from brain.links import *
import matplotlib.pyplot as plt
import time

start_time=time.strftime("%H:%M:%S")

# parameters
n_epochs = 1
coh_all=(100-np.logspace(np.log10(100),np.log10(3.125),num=6))/100
#coh_all = np.asarray([0.0,0.75])
rwrds = [0, 1, 0]#-0.02
i_class = [0, 1]
train_batch = 200000
test_batch = 20000
fix_length=3
trial_length=50
lr = 0.0001
gamma = 0.999
acc_all = np.zeros((12,6))
num_obs_all = np.zeros((12,6))

gpu=-1
for anum in xrange(20,28):
#np.random.seed(1)
    np.random.seed(anum)
    if anum>23:
        coh_all = np.asarray([0.0, 0.75])


    if gpu==1:
        chainer.cuda.get_device_from_id(1).use()

    for iii in xrange(1):
        # get training data - note that we select a subset of datapoints (n_samples is nr of samples per class)
        train_data = MNISTData(test=False, convolutional=False, n_samples=1000, classes = [0,1], centre=False)

        # define iterator
        data_iter = DataTaskMnist(train_data, batch_size=1, n_batches=train_batch, coh_all=coh_all, rewards=rwrds, noise_method=1,
                             fix_length=fix_length, trial_length=trial_length)

        val_data = MNISTData(test=True, convolutional=False, n_samples=1000, classes=[0,1], centre=False)
        val_iter = DataTaskMnist(val_data, batch_size=1, n_batches=test_batch, coh_all=coh_all, rewards=rwrds, noise_method=1,
                            fix_length=fix_length, trial_length=trial_length)

        # an actor-critic model assumes that the predictor's output is number of actions plus one for the value
        n_output = data_iter.n_output
        n_input = data_iter.n_input

        # define brain of agent
        model = ActorCriticModel(RNN2(n_input, n_output, n_hidden=100, link=GRU_custom))  # link=Elman))#L.StatefulGRU

        # define agent: only entropy on state actions
        agent = ActorCriticAgent(model, chainer.optimizers.Adam(alpha=lr), cutoff=train_batch, beta=1e-2, gamma=gamma, aac=True, gpu=gpu)  # Adam(alpha=0.0005)

        # add gradient clipping
        agent.optimizer.add_hook(chainer.optimizer.GradientClipping(1))

        # add monitors
        monitor0 = Oscilloscope(names=['p1', 'p2', 'p3'])
        agent.add_monitor(monitor0)
        monitor1=Oscilloscope(names=['cumulative reward'])
        agent.add_monitor(monitor1)
        monitorA=Monitor(names=['action'])
        agent.add_monitor(monitorA)
        monitor2=Monitor(names=['state','step','acc_trial'])
        data_iter.add_monitor(monitor2)

        monitor3 = Oscilloscope(names=['accuracy','resp_ratio','reward_rate'])
        data_iter.add_monitor(monitor3)
        agent.add_monitor(monitor3)

        # define world
        world = World(agent)

        # run world in training mode
        world.train(data_iter, n_epochs=n_epochs, plot=1000, monitor=1000, crit=False)

        fname = 'result/model_mnist22'+str(anum)+'_fl' + str(fix_length) + '_l' + str(trial_length) + '_g' + str(gamma).replace('.', '') + '_r' + str(rwrds[0]).replace('.', '')
        agent.model.save(fname)

        crw_log = monitor1['cumulative reward']
        acc_log = monitor2['acc_trial']
        Y = monitorA['action']
        p1 = monitor0['p1']
        p2 = monitor0['p2']
        p3 = monitor0['p3']

        np.savez(fname + '_log', crw_log, acc_log,Y,p1,p2,p3)

        coh_all = (100 - np.logspace(np.log10(100), np.log10(3.125), num=6)) / 100

        # run world in test mode
        val_data = MNISTData(test=True, convolutional=False, n_samples=1000, classes=[0,1], centre=False)

        val_iter = DataTaskMnist(val_data, batch_size=1, n_batches=test_batch, coh_all=coh_all, rewards=rwrds, noise_method=1,
                            fix_length=fix_length, trial_length=trial_length)

        # add monitor to model and iterator
        agent.add_monitor(Monitor(names=['action']))
        val_iter.add_monitor(Monitor(names=['state','step']))
        val_iter.add_monitor(Monitor(names=['coh']))

        monitor = Oscilloscope(names=['accuracy'])
        val_iter.add_monitor(monitor)
        agent.add_monitor(monitor)

        monitor_test = Oscilloscope(names=['p1', 'p2', 'p3','hidden-1','hidden-2'])
        agent.model.add_monitor(monitor_test)


        monitor = Monitor()
        agent.add_monitor(monitor)
        val_iter.add_monitor(monitor)

        # run in test mode
        world.test(val_iter, n_epochs=1, plot=0)


        # analysis
        num_obs = np.zeros(len(coh_all))
        bin_obs = np.zeros((1, 100))
        acc = np.zeros((len(coh_all), 100))
        acc1 = np.zeros(len(coh_all))

        nobs_all = np.zeros((1, 40))
        mobs_all = np.zeros((1, 40))

        # get variables
        Y = monitor['action']
        T = monitor['state']
        C1 = monitor['coh']

        p1 = monitor_test['p1']
        p2 = monitor_test['p2']
        p3 = monitor_test['p3']
        h1 = monitor_test['hidden-1']
        h2 = monitor_test['hidden-2']

        A = np.zeros(Y.shape)
        C = np.zeros((2, len(Y)))

        ### get time step index, class chosen
        temp = 0
        for ii in xrange(len(Y)):
            temp = temp + 1
            A[ii] = temp
            if Y[ii] != (data_iter.n_output - 1) or temp == trial_length:
                C[T[ii], ii] = temp
                temp = 0

        p1m = np.zeros((2, 40, 40))
        p2m = np.zeros((2, 40, 40))
        p3m = np.zeros((2, 40, 40))

        ### activity of output neurons
        for ic in xrange(2):
            for ii in xrange(40):
                idx = np.where(C[ic, :] == ii + 1)
                idx1 = []
                for i2 in xrange(ii + 1):
                    idx1 = [x - ii + i2 for x in list(idx)]
                    p1m[ic, ii, i2] = np.mean(p1[idx1])
                    p2m[ic, ii, i2] = np.mean(p2[idx1])
                    p3m[ic, ii, i2] = np.mean(p3[idx1])

        ### select trials at which a decision is being made
        idx = np.where(Y != data_iter.n_output - 1)[0]
        Y = Y[idx]
        T = T[idx]
        A = A[idx]
        C1 = C1[idx]

        ### numer of observations
        coh=np.unique(C1)
        for ii in xrange(len(coh)):
            idx1 = list(np.where(C1 == coh[ii]))
            num_obs[ii] = np.mean(A[idx1])

        ### correct trials
        B = np.equal(Y, T)


        ### accuracies
        for ii in xrange(len(coh)):
            idx1 = list(np.where(C1 == coh[ii]))
            acc1[ii] = np.sum(B[idx1]) / float(len(idx1[0]))
            for iii in xrange(50):
                idx2 = list(np.where((A == iii + 1) & (C1 == coh[ii])))
                acc[ii,iii] = np.sum(B[idx2]) / float(len(idx2[0]))

        ### hidden state activities
        FR = np.zeros((2,len(coh_all),100))
        for iiii in xrange(2):
            for iii in xrange(len(coh_all)):
                for ii in xrange(100):
                    idx = np.where((A == ii+1) & (C1 == coh_all[iii]) & (T == iiii))
                    FR[iiii,iii,ii] = np.mean(h1[idx,:])

        plt.close("all")

        print(num_obs)
        print(acc1)
        acc_all[anum-20,:]=acc1
        num_obs_all[anum-20,:]=num_obs

print(acc_all)
print(num_obs_all)
