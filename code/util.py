import numpy as np
from numpy.random import choice
from matplotlib import pyplot as plt
import GPy
from scipy.stats import norm, multivariate_normal
from scipy.stats import truncnorm

def setting_parameters(f_num=3, c_num=2):
    kernel = GPy.kern.RBF(f_num, variance = 1, lengthscale = 0.4)
    lik = GPy.likelihoods.Bernoulli()
    parameters_ = [f_num, c_num, kernel, lik]
    return parameters_

def setting_parameters_regression(f_num=3, c_num=2):
    kernel = GPy.kern.RBF(f_num, variance = 1, lengthscale = 0.4)
    lik = GPy.likelihoods.gaussian.Gaussian(variance=1e-4)
    parameters_ = [f_num, c_num, kernel, lik]
    return parameters_

def setting_px(f_num, xinterval):
    px = 1/(xinterval[1] - xinterval[0])**f_num
    px_log = -f_num*np.log(xinterval[1] - xinterval[0])
    return px, px_log

# discrete_label = False
# optimize_label = True
#
# Perror = 0.2

def Xtruncated(xlower, xupper, xspace, f_num):
    idxarray = np.all(xspace >= xlower, axis=1) & np.all(xspace <= xupper, axis = 1)
    return xspace[idxarray].reshape(-1, f_num), idxarray



def ModelDraw(model, name, data):
    #this is for f_num == 2
    pass
    ax = plt.subplot(projection='ternary')
    xspace = model.XspaceGenerate(100000, data.xspace())
    pyTheta = model.predict_proba(xspace)
    c_f = ax.tricontourf(xspace[:, 0], xspace[:, 1], xspace[:, 2], pyTheta[:, -1])
    c_list = []
    for type in model.gpc.Y:
        c_list.append('r' if type[0] > 0.5 else 'b')
    ax.scatter(model.gpc.X[:, 0], model.gpc.X[:, 1], model.gpc.X[:, 2], c=c_list)
    plt.colorbar(c_f)
    plt.savefig('c_'+name)
    plt.show()


