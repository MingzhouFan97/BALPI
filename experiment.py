import dataset
import surrogate
import util
import numpy as np
import matplotlib.pyplot as plt
position = 'tick1'
class Experiment():
    def __init__(self, data_all, f_num, c_num):
        self.data_all = data_all
        self.f_num = f_num
        self.c_num = c_num
        self.model = None
        self.error = []
    def initialize(self, init_num, given_init=None):
        pass

    def perform_one_iteration(self, retrain):
        xstar, max_value = self.optimizer(util=self.utility, model=self.model, data=self.data_all)
        f = lambda x: dataset.GroundTruthFunction(x, self.data_all)
        y_st = f(xstar[np.newaxis, :])
        ystar = y_st[0, 0] if len(y_st.shape)==2 else y_st[0]
        self.model.Update(xstar, ystar, optimize=retrain)

    def error_recording(self):
        return self.error
    def generate_figure(self, dirn, name):
        pass

class Experiement_fixed_discrete_data(Experiment):
    def __init__(self, data_all, f_num, c_num, hyper_int, optimizer, utility):
        super().__init__(data_all, f_num, c_num)
        self.hyper_int = hyper_int
        self.optimizer = optimizer
        self.utility = utility

    def initialize(self, init_num, given_init=None):
        if given_init is None:
            f = lambda x: dataset.GroundTruthFunction(x, self.data_all)
            X_, Y_, Xindex = dataset.InitialDataGenerator(f, self.f_num, init_num, xspace=self.data_all.xspace(), x_int=self.data_all.x_int)
        else:
            X_, Y_, Xindex = given_init[0], given_init[1], given_init[2]
        if not np.sum(Y_):
            X_ = np.concatenate([X_, np.array([[100.]*self.f_num])], axis=0)
            Y_ = np.concatenate([Y_, np.array([[1]])], axis=0)
        self.model = surrogate.Model(X_, Y_, parameters=util.setting_parameters(f_num=self.f_num, c_num=self.c_num), optimize=True, xinterval=self.data_all.x_int, hyper_inteval=self.hyper_int)
        self.model.dataidx = Xindex
        self.error.append(self.model.ObcClassifierError(10000, self.data_all))
    def perform_one_iteration(self, retrain):
        super().perform_one_iteration(retrain)
        self.error.append(self.model.ObcClassifierError(10000, self.data_all))

    def generate_figure(self, dirn, name):
        self.ModelDraw(self.model, dirn + name, data=self.data_all)

    def ModelDraw(self, model, name, data):
        # this is for f_num == 2
        ax = plt.subplot(projection='ternary')
        xspace = model.XspaceGenerate(100000, data.xspace())
        pyTheta = model.predict_proba(xspace)
        if xspace.shape[1] == 3:
            c_f = ax.tricontourf(xspace[:, 0], xspace[:, 1], xspace[:, 2], pyTheta[:, -1]>0.5)
        else:
            c_f = ax.tricontourf(xspace[:, 0], xspace[:, 1], 1 - xspace[:, 0] - xspace[:, 1], pyTheta[:, -1]>0.5)
        c_list = []
        for type in model.gpc.Y:
            c_list.append('r' if type[0] > 0.5 else 'b')
        if xspace.shape[1] == 3:
            ax.scatter(model.gpc.X[:, 0], model.gpc.X[:, 1], model.gpc.X[:, 2], c=c_list)
        else:
            ax.scatter(model.gpc.X[:, 0], model.gpc.X[:, 1], 1 - model.gpc.X[:, 0] - model.gpc.X[:, 1], c=c_list)
        plt.colorbar(c_f, shrink=0.8)
        ax.taxis.set_ticks_position(position)
        ax.laxis.set_ticks_position(position)
        ax.raxis.set_ticks_position(position)
        ax.taxis.set_label_position(position)
        ax.laxis.set_label_position(position)
        ax.raxis.set_label_position(position)
        ax.set_tlabel('NiTi')
        ax.set_llabel('Cu')
        ax.set_rlabel('Hf')
        plt.savefig(name)
        plt.clf()



class Experiement_fixed_discrete_data_LP(Experiment):
    def __init__(self, data_all, f_num, c_num, hyper_int, optimizer, utility):
        super().__init__(data_all, f_num, c_num)
        self.hyper_int = hyper_int
        self.optimizer = optimizer
        self.utility = utility

    def initialize(self, init_num, given_init=None):
        if given_init is None:
            f = lambda x: dataset.GroundTruthFunction(x, self.data_all)
            X_, Y_, Xindex = dataset.InitialDataGenerator(f, self.f_num, init_num, xspace=self.data_all.xspace(), x_int=self.data_all.x_int)
        else:
            X_, Y_, Xindex = given_init[0], given_init[1], given_init[2]
        if not np.sum(Y_):
            X_ = np.concatenate([X_, np.array([[100.]*self.f_num])], axis=0)
            Y_ = np.concatenate([Y_, np.array([[1]])], axis=0)
        self.model = surrogate.Model_LP(X_, Y_, parameters=util.setting_parameters(f_num=self.f_num, c_num=self.c_num))
        self.model.dataidx = Xindex
        self.error.append(self.model.ObcClassifierError(10001, self.data_all))
    def perform_one_iteration(self, retrain):
        super().perform_one_iteration(retrain)
        self.error.append(self.model.ObcClassifierError(10001, self.data_all))

    def generate_figure(self, dirn, name):
        self.ModelDraw(self.model, dirn + name, data=self.data_all)

    def ModelDraw(self, model, name, data):
        # this is for f_num == 2
        ax = plt.subplot(projection='ternary')
        xspace = model.XspaceGenerate(100000, data.xspace())
        pyTheta = model.predict_proba(xspace)
        if xspace.shape[1] == 3:
            c_f = ax.tricontourf(xspace[:, 0], xspace[:, 1], xspace[:, 2], pyTheta[:, -1]>0.5)
        else:
            c_f = ax.tricontourf(xspace[:, 0], xspace[:, 1], 1-xspace[:, 0]-xspace[:, 1], pyTheta[:, -1]>0.5)
        c_list = []
        for type in model.Y:
            c_list.append('r' if type[0] > 0.5 else 'b')
        if xspace.shape[1] == 3:
            ax.scatter(model.X[:, 0], model.X[:, 1], model.X[:, 2], c=c_list)
        else:
            ax.scatter(model.X[:, 0], model.X[:, 1], 1-model.X[:, 0]-model.X[:, 1], c=c_list)

        ax.set_tlabel('NiTi')
        ax.set_llabel('Cu')
        ax.set_rlabel('Hf')
        plt.colorbar(c_f, shrink=0.8)
        ax.taxis.set_ticks_position(position)
        ax.laxis.set_ticks_position(position)
        ax.raxis.set_ticks_position(position)
        ax.taxis.set_label_position(position)
        ax.laxis.set_label_position(position)
        ax.raxis.set_label_position(position)
        plt.savefig(name)
        plt.clf()

class Experiement_fixed_discrete_data_LS(Experiment):
    def __init__(self, data_all, f_num, c_num, hyper_int, optimizer, utility):
        super().__init__(data_all, f_num, c_num)
        self.hyper_int = hyper_int
        self.optimizer = optimizer
        self.utility = utility

    def initialize(self, init_num, given_init=None):
        if given_init is None:
            f = lambda x: dataset.GroundTruthFunction(x, self.data_all)
            X_, Y_, Xindex = dataset.InitialDataGenerator(f, self.f_num, init_num, xspace=self.data_all.xspace(), x_int=self.data_all.x_int)
        else:
            X_, Y_, Xindex = given_init[0], given_init[1], given_init[2]
        if not np.sum(Y_):
            X_ = np.concatenate([X_, np.array([[100.]*self.f_num])], axis=0)
            Y_ = np.concatenate([Y_, np.array([[1]])], axis=0)
        self.model = surrogate.Model_LS(X_, Y_, parameters=util.setting_parameters(f_num=self.f_num, c_num=self.c_num))
        self.model.dataidx = Xindex
        self.error.append(self.model.ObcClassifierError(10001, self.data_all))
    def perform_one_iteration(self, retrain):
        super().perform_one_iteration(retrain)
        self.error.append(self.model.ObcClassifierError(10001, self.data_all))

    def generate_figure(self, dirn, name):
        self.ModelDraw(self.model, dirn + name, data=self.data_all)

    def ModelDraw(self, model, name, data):
        # this is for f_num == 2
        ax = plt.subplot(projection='ternary')
        xspace = model.XspaceGenerate(100000, data.xspace())
        pyTheta = model.predict_proba(xspace)
        if xspace.shape[1] == 3:
            c_f = ax.tricontourf(xspace[:, 0], xspace[:, 1], xspace[:, 2], pyTheta[:, -1]>0.5)
        else:
            c_f = ax.tricontourf(xspace[:, 0], xspace[:, 1], 1 - xspace[:, 0] - xspace[:, 1], pyTheta[:, -1]>0.5)
        c_list = []
        for type in model.Y:
            c_list.append('r' if type[0] > 0.5 else 'b')
        if xspace.shape[1] == 3:
            ax.scatter(model.X[:, 0], model.X[:, 1], model.X[:, 2], c=c_list)
        else:
            ax.scatter(model.X[:, 0], model.X[:, 1], 1 - model.X[:, 0] - model.X[:, 1], c=c_list)
        plt.colorbar(c_f, shrink=0.8)
        ax.taxis.set_ticks_position(position)
        ax.laxis.set_ticks_position(position)
        ax.raxis.set_ticks_position(position)
        ax.taxis.set_label_position(position)
        ax.laxis.set_label_position(position)
        ax.raxis.set_label_position(position)
        ax.set_tlabel('NiTi')
        ax.set_llabel('Cu')
        ax.set_rlabel('Hf')
        plt.savefig(name)
        plt.clf()


class Experiement_fixed_discrete_data_regression(Experiment):
    def __init__(self, data_all, f_num, c_num, hyper_int, optimizer, utility, threshold):
        super().__init__(data_all, f_num, c_num)
        self.hyper_int = hyper_int
        self.optimizer = optimizer
        self.utility = utility
        self.threshold = threshold

    def initialize(self, init_num, given_init=None):
        if given_init is None:
            f = lambda x: dataset.GroundTruthFunction(x, self.data_all)
            X_, Y_, Xindex = dataset.InitialDataGenerator(f, self.f_num, init_num, xspace=self.data_all.xspace(), x_int=self.data_all.x_int)
        else:
            X_, Y_, Xindex = given_init[0], given_init[1], given_init[2]
        if not np.sum(Y_):
            X_ = np.concatenate([X_, np.array([[100.]*self.f_num])], axis=0)
            Y_ = np.concatenate([Y_, np.array([[1]])], axis=0)
        self.model = surrogate.Model_regression(X_, Y_, parameters=util.setting_parameters_regression(f_num=self.f_num, c_num=self.c_num), optimize=True, xinterval=self.data_all.x_int, hyper_inteval=self.hyper_int)
        self.model.dataidx = Xindex
        self.error.append(self.model.ObcClassifierError(10001, self.data_all, self.threshold))

    def perform_one_iteration(self, retrain):
        super().perform_one_iteration(retrain)
        self.error.append(self.model.ObcClassifierError(10001, self.data_all, self.threshold))
    def generate_figure(self, dirn, name, threshold=.8):
        self.ModelDraw(self.model, dirn + name, data=self.data_all, threshold=threshold)

    def ModelDraw(self, model, name, data, threshold):
        # this is for f_num == 2
        ax = plt.subplot(projection='ternary')
        xspace = model.XspaceGenerate(100000, data.xspace())
        pyTheta = model.predict_proba(xspace)[:, 0:1]
        if xspace.shape[1] == 3:
            c_f = ax.tricontourf(xspace[:, 0], xspace[:, 1], xspace[:, 2], pyTheta[:, -1]>0.8)
        else:
            c_f = ax.tricontourf(xspace[:, 0], xspace[:, 1], 1 - xspace[:, 0] - xspace[:, 1], pyTheta[:, -1]>0.8)
        c_list = []
        for type in model.gpc.Y:
            c_list.append('r' if type[0] > threshold else 'b')
        if xspace.shape[1] == 3:
            ax.scatter(model.gpc.X[:, 0], model.gpc.X[:, 1], model.gpc.X[:, 2], c=c_list)
        else:
            ax.scatter(model.gpc.X[:, 0], model.gpc.X[:, 1], 1 - model.gpc.X[:, 0] - model.gpc.X[:, 1], c=c_list)
        plt.colorbar(c_f, shrink=0.8)
        ax.taxis.set_ticks_position(position)
        ax.laxis.set_ticks_position(position)
        ax.raxis.set_ticks_position(position)
        ax.taxis.set_label_position(position)
        ax.laxis.set_label_position(position)
        ax.raxis.set_label_position(position)
        ax.set_tlabel('NiTi')
        ax.set_llabel('Cu')
        ax.set_rlabel('Hf')
        plt.savefig(name)
        plt.clf()
