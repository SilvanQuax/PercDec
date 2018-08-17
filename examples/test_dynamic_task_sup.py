# test object categorization under noise as an RL task
import sys

sys.path.append('/home/squax/GitHub/CMS/')
from agent.supervised import *
from brain.monitor import *
from brain.models import Classifier
from brain.networks import *
from world.base import World
from world.data import *
from world.tasks import *
from brain.links import *
import matplotlib.pyplot as plt
import tools

if len(sys.argv)>1:
    hidden=int(sys.argv[1])
    pn=float(sys.argv[2])
else:
    hidden=3
    pn=(100-np.logspace(np.log10(100),np.log10(3.125),num=6))/100




# parameters
n_epochs = 1
# n_level = [0,0.75,0.85,0.9,0.92,0.95] #[0,0.4,0.6,0.8,0.9]
n_level = [0]
noise_all=pn# [1 - x for x in [0.08, 0.15, 0.25, 0.5]]
rwrds = [0, 1, -1]
i_class = [0, 1]
n_steps=10



num_obs = np.zeros(len(noise_all))
bin_obs = np.zeros((len(n_level), 100))
acc = np.zeros((len(noise_all),n_steps))
acc1 = np.zeros(len(noise_all))

nobs_all = np.zeros((len(n_level), 40))
mobs_all = np.zeros((len(n_level), 40))

#chainer.cuda.get_device_from_id(0).use()

for t_steps in xrange(1,n_steps+1):
    train_batch = 20000*t_steps
    test_batch = 10000*t_steps
    ### get training data - note that we select a subset of datapoints (n_samples is nr of samples per class)
    #train_data = ClutteredMNIST(test=False, convolutional=True, n_samples=10000, classes = None)
    train_data = MNISTData(test=False, convolutional=False, n_samples=10000, classes=None)
    ### define iterator
    data_iter = DynamicSupervised(train_data, noise_all=noise_all,batch_size=100, n_batches = train_batch, noise_method=1, cnoise=True)

    ###for validation
    # val_data = MNISTData(test=True, convolutional=False, n_samples=1000, classes=None)
    # val_iter = DynamicSupervised(val_data, noise_all=noise_all,batch_size=10, n_batches = test_batch, noise_method=1, cnoise=True)

    ### an actor-critic model assumes that the predictor's output is number of actions plus one for the value
    n_output = data_iter.n_output

    ### define brain of agent
    model = Classifier(RNN(data_iter.n_input, n_output, n_hidden=hidden, link=GRU_custom2, n_hidden_layers=1),gpu=-1) #link=Elman))#L.StatefulGRU
    #model = Classifier(RNNFBI(data_iter.n_input, n_output, n_hidden=100)) #link=Elman))#L.StatefulGRU

    #model = Classifier(CRNN(train_data._n_input, n_output, n_hidden1=32, n_hidden2=50, link=L.StatefulGRU)).to_gpu() #link=Elman))#L.StatefulGRU
    #model = Classifier(MLP(data_iter.n_input, n_output, n_hidden=50)) #link=Elman))#L.StatefulGRU
    #model = Classifier(CRNN2(train_data._n_input, n_output, n_hidden=32)).to_gpu() #link=Elman))#L.StatefulGRU

    ### define agent
    agent = StatefulAgent(model, chainer.optimizers.Adam(alpha=0.005), gpu=-1, cutoff=t_steps, last=True)#Adam(alpha=0.0005)
    ### add gradient clipping
    #agent.optimizer.add_hook(chainer.optimizer.GradientClipping(1))


    ### add oscilloscope
    data_iter.add_monitor(Monitor(names=['state']))

    agent.add_monitor(Oscilloscope(names=['accuracy']))
    ### define world
    world = World(agent)

    fname='mnist_sup_cn3_h' + str(hidden) + '_t' + str(t_steps)
    print(fname)

    ### run world in training mode
    world.train(data_iter, n_epochs=n_epochs, plot=1000, monitor=1000)
    #world.validate(data_iter, validation=val_iter, n_epochs=n_epochs, plot=1000, monitor=1000)

    model.save('result/supervised/model_'+fname)

    ### run world in test mode

    #val_data = ClutteredMNIST(test=True, convolutional=True, n_samples=1000, classes = None)
    val_data = MNISTData(test=True, convolutional=False, n_samples=1000, classes=None)
    # define iterator
    val_iter = DynamicSupervised(val_data, noise_all=noise_all,batch_size=1, n_batches = test_batch, noise_method=1, cnoise=True)


    # add monitor to model and iterator
    val_iter.add_monitor(Monitor(names=['state']))
    val_iter.add_monitor(Monitor(names=['noise']))

    # val_iter.add_monitor(Monitor(names=['nobs']))

    monitor = Monitor()
    agent.add_monitor(monitor)
    val_iter.add_monitor(monitor)

    # run in test mode
    world.test(val_iter, n_epochs=1, plot=0)
    #print(agent.model.predictor[0][0].W.W)
    # get variables
    T = monitor['state']
    Y = monitor['prediction']
    C1 = monitor['noise']

    A = np.zeros(Y.shape)
    C = np.zeros((2, len(Y)))

    A=np.zeros(Y.shape)
    temp=0
    for ii in xrange(len(Y)):
        temp = temp + 1
        A[ii]=temp
        if temp==t_steps:
            temp = 0

    coh=np.unique(C1)
    for ii in xrange(len(coh)):
        idx1 = list(np.where(C1 == coh[ii]))
        num_obs[ii] = np.mean(A[idx1])

    # num_obs[iii] = test_batch / float(len(A))
    # bin_obs[iii] = np.bincount(A.astype('int32'), minlength=200)

    B = np.equal(Y, T)



    # for ii in xrange(len(coh)):
    #     idx1 = list(np.where(C1 == coh[ii]))
    #     acc1[ii] = np.sum(B[idx1]) / float(len(idx1[0]))
    #     idx2 = list(np.where((A == t_steps) & (C1 == coh[ii])))
    #     acc[ii,t_steps] = np.sum(B[idx2]) / float(len(idx2[0]))

    for ii in xrange(len(coh)):
        idx1 = list(np.where(C1 == coh[ii]))
        acc1[ii] = np.sum(B[idx1]) / float(len(idx1[0]))

        idx2 = list(np.where((A == t_steps) & (C1 == coh[ii])))
        acc[ii, t_steps-1] = np.sum(B[idx2]) / float(len(idx2[0]))

            # idx1[:]=[x -ii for x in idx1]
        # nobs_all[iii, ii] = np.mean(nobs[idx1])
        # mobs_all[iii, ii] = np.mean(mobs[idx1])

    #### hidden state activities
    #print(num_obs)
    print(acc)
    print(acc1)
    plt.close("all")

np.savetxt('result/supervised/'+ fname + '_class_accuracy', acc, fmt='%.4f', delimiter=',')

    # print(nobs_all[iii,:5])
    # print(mobs_all[iii,:5])
# print('reset=every')
# print(n_level)
# print(num_obs)
# print(bin_obs)
# print(acc)
# print(p1m[0, :10, :10])
# print(p2m[0, :10, :10])
# print(p3m[0, :10, :10])
# print(p1m[1, :10, :10])
# print(p2m[1, :10, :10])
# print(p3m[1, :10, :10])
