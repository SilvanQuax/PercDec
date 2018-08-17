from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from chainer import cuda

#####
## Monitor base class

class Monitor(object):

    def __init__(self, names=None, len=np.inf):
        """

        :param names: if defined then the name or list of names indicates keys to store
        :param len: determines the length of the monitor; stores the last len items at most
        """

        self.dict = defaultdict(list)

        self.names = [names] if names is str else names

        self.len = len

    def reset(self):
        self.dict = defaultdict(list)

    def set(self, name, value):
        """Monitor cannot make shallow copies since we use it to store chainer Variables as well for RL buffering.
           Make an explicit copy of any variable you wish to store without having call by reference behaviour

        :param name: dictionary key
        :param value: dictionary value
        :return:
        """

        if self.names is None or name in self.names:
            if self.len==1:
                self.dict[name] = [value]
            else:
                if self.dict[name] == self.len:
                    self.dict[name] = self.dict[name][1:].append(value)
                else:
                    self.dict[name].append(value)

    def keys(self):
        return self.dict.keys()

    def __getitem__(self, item):
        """Returns dict[name] as numpy array

        :param item: key name
        :return: numpy array
        """

        assert(item in self.keys())
        data = np.asarray(self.dict[item])
        if data.ndim <=2:
            return data.reshape([np.prod(data.shape)], order='F')
        else:
            return data.reshape([np.prod(data.shape[:2])] + list(data.shape[2:]), order='F')

    def run(self):
        """Run a function on the fly; implemented by inheriting from this monitor

        :return:
        """
        pass


class Oscilloscope(Monitor):
    """
    Defines an oscilloscope for one or more signals of interest defined in names
    """

    def __init__(self, names=None, len=np.inf, ylabel=None):

        super(Oscilloscope, self).__init__(names, len)
        self.fig = self.ax = self.hl = None

        self.ylabel = ylabel


    def run(self):

        # set a separate figure for each oscilloscope

        if self.names is None:
            keys = self.keys()
        else:
            keys = self.names

        if self.fig is None:

            self.fig, self.ax = plt.subplots()
            self.ax.set_xlabel('t')
            if self.ylabel:
                self.ax.set_ylabel(self.ylabel)
            self.hl = np.empty(len(keys), dtype=object)
            for i in range(len(keys)):
                key = keys[i]
                self.hl[i], = self.ax.plot(np.arange(self[key].size), self[key],'o')
            self.fig.legend(self.hl, tuple(keys))
            self.fig.show()

        else:

            for i in range(len(keys)):
                key = keys[i]
                self.hl[i].set_xdata(np.arange(self[key].size))
                self.hl[i].set_ydata(self[key])
            self.fig.legend(self.hl, tuple(keys))
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def save(self, fname):
        self.fig.savefig(fname)
        plt.close()