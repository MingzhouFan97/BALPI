import numpy as np
import matplotlib.pyplot as plt
from numpy.random import choice
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import mpltern



def generate_grid_force(dim, res):
    candidate = []
    for d in range(dim - 1):
        if len(candidate) == 0:
            candidate = [[res*i] for i in range(int(1/res) + 1)]
        else:
            subcandidate = []
            for entry in candidate:
                remain = 1-sum(entry)
                subcandidate += [entry + [res*i] for i in range(int(remain/res) + 1)]
            candidate = subcandidate
    for i, entry in enumerate(candidate):
        candidate[i].append(1 - sum(entry))
    return np.array(candidate)


def generate_fake_data(candidate):
    res = ((candidate[:, 2]/2)**2 + ((candidate[:, 0]-0.3)/0.2)**2) < .1
    return res

class DataSet(object):
    def __init__(self):
        pass
    def query(self, x):
        pass
    def x_dim(self):
        pass
    def set_x_int(self, x_int):
        self.x_int = x_int



class DataSet_from_csv(DataSet):# Store the data as a csv file whose last column is target and the rest are input domain
    def __init__(self, f_name):
        super().__init__()
        self.data = self.preprocess(np.genfromtxt(f_name, delimiter=','))
    def preprocess(self, np_array):
        return np_array
    def generate_plt(self):
        ax = plt.subplot(projection='ternary')
        c_f = ax.tricontourf(self.data[:, 0], self.data[:, 1], self.data[:, 2], self.data[:, 3])
        plt.colorbar(c_f)
        # plt.legend()
        return plt
    def xspace(self):
        return self.data[:, :-1]
    def query(self, x):
        if x.shape[0] > 1:
            ind = []
            for i in range(x.shape[0]):
                ind.append((abs(self.data[:, :-1] - x[i:i+1, :]) < 1e-5).all(axis=1).nonzero())
            ind = np.array(ind).reshape(-1)
        else:
            ind = (abs(self.data[:, :-1] - x) < 1e-5).all(axis=1).nonzero()
        return self.data[ind, -1]


class DataSet_from_csv_boundary(DataSet):  # Store the data as a csv file whose last column is target and the rest are input domain
    def __init__(self, f_name, type=0):
        super().__init__()
        if type == 0:
            self.data = self.preprocess_1(np.genfromtxt(f_name, delimiter=',', skip_header=0, dtype=None))
            data_all = np.random.rand(100000, 2)
            self.data_all = data_all[data_all[:, 0] + data_all[:, 1] <= 1]
        if type == 1:
            self.data = self.preprocess_2(np.genfromtxt(f_name, delimiter=',', skip_header=0, dtype=None))
        self.polygon = Polygon(self.data)


    def preprocess_1(self, np_array):
        return np_array[:, :2]

    def preprocess_2(self, np_array):
        np_array[:, 1] = np_array[:, 1] / (1 - np_array[:, 0])
        return np_array[:, :2]

    def query_one(self, x):
        point = Point(x)
        return self.polygon.contains(point)

    def query(self, x):
        res = [self.query_one(x_i) for x_i in x]
        return np.array(res, dtype=int)

    def xspace(self):
        return self.data_all[:, :]


def GroundTruthFunction(x, data_d):
    # x is a single point f_num
    y = data_d.query(x)
    return y


def XspaceGenerate_(x_num, f_num, xinterval=None, xspace=None):
    if xspace is not None:
        if x_num >= len(xspace):
            return xspace
        sampleidx = choice(range(xspace.shape[0]), x_num, replace = False)#############################
        return xspace[sampleidx]
    print((x_num, f_num))
    print(xinterval[0], xinterval[1], (x_num, f_num))
    xspace = np.random.uniform(xinterval[0], xinterval[1], (x_num, f_num))
    return xspace

def InitialDataGenerator(f, f_num, initial_num=10, xspace=None, x_int=None):
    X_ = XspaceGenerate_(initial_num, f_num, x_int, xspace)
    Y_ = np.zeros((initial_num, 1))
    for i in range(initial_num):
        Y_[i] = f(X_[i:i + 1])
    Xindex = None
    return X_, Y_, Xindex


if __name__ == "__main__":
    X_truth = generate_grid_force(3, 0.01)
    Y_truth = generate_fake_data(X_truth)

    data = np.concatenate([X_truth, Y_truth[:, np.newaxis]], axis=1)
    print(data)
    np.savetxt("./TC_data/toy_data.csv", data, delimiter=",")

    ax = plt.subplot(projection='ternary')

    c_f = ax.tricontourf(X_truth[:, 0], X_truth[:, 1], X_truth[:, 2], Y_truth)
    plt.colorbar(c_f)
    # plt.legend()
    plt.show()
    # data_read('./TC_data/fake.csv')