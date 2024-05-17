# -*- coding: utf-8 -*-
# %%
"""
Created on Fri Sep 18 00:03:35 2020

@author: Aggie


continuous code
"""

import numpy as np
from numpy.random import choice
from matplotlib import pyplot as plt
import GPy
from scipy.stats import norm, multivariate_normal
from scipy.stats import truncnorm
from scipy import special
from scipy import integrate
import math
import sys, os
import util
from sklearn.semi_supervised import LabelSpreading, LabelPropagation

sys.path.append(os.path.dirname(sys.path[0]))
import torch

A = np.polynomial.hermite.hermgauss(8)

def GroundTruthFunction(x, data_d):
    # x is a single point f_num
    y = data_d.query(x)
    return y


class Model():

    def __init__(self, X, Y, parameters=None,
                 optimize=False, xinterval=(0., 1.), hyper_inteval= [[1e-4, .15], [1e-4, 1000.]]):  #####################################################
        # __slots__ = ['parameters', 'f_num', 'gpc', 'xinterval', 'optimize', 'lik', 'c_num']
        # X size is x_num*fnum
        # Y size is x_num
        if (parameters is None) or (xinterval is None):
            raise('ERROR!!!!!!!!!!!!!!!!!')
        self.parameters = parameters
        self.f_num = parameters[0]
        self.c_num = parameters[1]
        self.kernel = parameters[2]
        self.lik = parameters[3]
        self.hyper_inteval = hyper_inteval
        m = GPy.core.GP(X=X,
                        Y=Y,
                        kernel=self.kernel,
                        inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                        likelihood=self.lik)
        if optimize:
            m.optimize_restarts(optimizer='bfgs', num_restarts=40, max_iters=2000, verbose=False)
        self.gpc = m
        self.check_hyper_boundary()
        self.xinterval = xinterval
        self.optimize = optimize

    def check_hyper_boundary(self):
        if self.hyper_inteval is None:
            raise('!!!!!!!!')
        if self.gpc.kern.lengthscale.item() < self.hyper_inteval[0][0]:
            self.kernel = GPy.kern.RBF(self.f_num, variance=self.gpc.kern.variance.item(), lengthscale=self.hyper_inteval[0][0])
            self.gpc = GPy.core.GP(X=self.gpc.X,
                            Y=self.gpc.Y,
                            kernel=self.kernel,
                            inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                            likelihood=self.lik)
        elif self.gpc.kern.lengthscale.item() > self.hyper_inteval[0][1]:
            self.kernel = GPy.kern.RBF(self.f_num, variance=self.gpc.kern.variance.item(),
                                  lengthscale=self.hyper_inteval[0][1])
            self.gpc = GPy.core.GP(X=self.gpc.X,
                                   Y=self.gpc.Y,
                                   kernel=self.kernel,
                                   inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                                   likelihood=self.lik)

        if self.gpc.kern.variance.item() < self.hyper_inteval[1][0]:
            self.kernel = GPy.kern.RBF(self.f_num, variance=self.hyper_inteval[1][0],
                                       lengthscale=self.gpc.kern.lengthscale.item())
            self.gpc = GPy.core.GP(X=self.gpc.X,
                                   Y=self.gpc.Y,
                                   kernel=self.kernel,
                                   inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                                   likelihood=self.lik)
        elif self.gpc.kern.variance.item() > self.hyper_inteval[1][1]:
            self.kernel = GPy.kern.RBF(self.f_num, variance=self.hyper_inteval[1][1],
                                       lengthscale=self.gpc.kern.lengthscale.item())
            self.gpc = GPy.core.GP(X=self.gpc.X,
                                   Y=self.gpc.Y,
                                   kernel=self.kernel,
                                   inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                                   likelihood=self.lik)
    def predict_proba(self, x):
        x = x.reshape(-1, self.f_num)
        M = len(x) // 1000
        if M <= 1:
            py_1 = self.gpc.predict(x)[0]
            py_0 = 1 - py_1
            pymat = np.concatenate((py_0, py_1), axis=1)  # pymat size x_num*cnum
            return pymat
        else:
            pymat1 = np.zeros((len(x), 2))
            for m in range(M):
                idx = range(m * 1000, m * 1000 + 1000)
                pymat1[idx, 1:2] = self.gpc.predict(x[idx, :])[0]
                pymat1[idx, 0] = 1 - pymat1[idx, 1]
            idx = range(m * 1000 + 1000, len(x))
            pymat1[idx, 1:2] = self.gpc.predict(x[idx, :])[0]
            pymat1[idx, 0] = 1 - pymat1[idx, 1]

            return pymat1

    def _noiseless_predict_torch(self, xt):
        woodbury_inv = torch.tensor(self.gpc.posterior.woodbury_inv)
        woodbury_vector = torch.tensor(self.gpc.posterior.woodbury_vector)
        X_ = torch.tensor(self.gpc.X)
        K = self.K

        mu_t = K(xt, X_) @ woodbury_vector
        sigma_tt = K(xt, xt) - K(xt, X_) @ woodbury_inv @ K(X_, xt)

        return mu_t, sigma_tt

    def K(self, xt, xs):

        kern = self.gpc.kern
        assert (kern.name == 'rbf')  # the function is only coded for rbf kernel

        l1 = kern.lengthscale.item()
        l2 = kern.variance.item()
        Kts = l2 * torch.exp(-torch.cdist(xt, xs) ** 2 / 2 / l1 ** 2)
        return Kts

    def predict_proba_torch(self, xt):
        xt = xt.reshape(-1, self.f_num)
        assert (type(xt) == torch.Tensor)

        Phi = lambda x: 0.5 * (torch.erf(x / math.sqrt(2)) + 1)
        mu_t, sigma_tt = self._noiseless_predict_torch(xt)

        ft_hat = mu_t / torch.sqrt(sigma_tt + 1)
        py_1 = Phi(ft_hat)
        py_0 = 1 - py_1
        pymat = torch.cat([py_0, py_1], axis=1)
        return pymat

    def _calculate_mean_and_variance(self, xt, xs):
        xt = xt.reshape(-1, self.f_num)
        xs = xs.reshape(-1, self.f_num)
        muvar = self.gpc.predict_noiseless(np.concatenate((xs, xt)), full_cov=False)
        mu = muvar[0].reshape(-1)
        mu_s = mu[0:-1]
        mu_t = mu[-1]

        var = muvar[1].reshape(-1)
        sigma_ss = var[0:-1]
        X_ = self.gpc.X
        sigma_st = self.gpc.kern.K(xs, xt) - self.gpc.kern.K(xs,
                                                             X_) @ self.gpc.posterior.woodbury_inv @ self.gpc.kern.K(X_,
                                                                                                                     xt)
        sigma_st = sigma_st.reshape(-1)
        sigma_tt = var[-1]
        sigma_tt_hat = sigma_tt - sigma_st ** 2 / sigma_ss
        return mu_s, mu_t, sigma_ss, sigma_st, sigma_tt_hat

    def _calculate_mean_and_variance_torch(self, x1, x2):
        xt = x1.reshape(-1, self.f_num)
        xs = torch.tensor(x2.reshape(-1, self.f_num))

        X_ = torch.tensor(self.gpc.X)
        woodbury_inv = torch.tensor(self.gpc.posterior.woodbury_inv)
        woodbury_vector = torch.tensor(self.gpc.posterior.woodbury_vector)

        muvar = self.gpc.predict_noiseless(x2, full_cov=False)
        mu_s = torch.tensor(muvar[0])
        sigma_ss = torch.tensor(muvar[1])

        K = self.K
        mu_t, sigma_tt = self._noiseless_predict_torch(xt)
        sigma_st = K(xs, xt) - K(xs, X_) @ woodbury_inv @ K(X_, xt)
        sigma_tt_hat = sigma_tt - sigma_st ** 2 / sigma_ss
        mu_s = mu_s.reshape(-1)
        mu_t = mu_t.reshape(-1)
        sigma_ss = sigma_ss.reshape(-1)
        sigma_st = sigma_st.reshape(-1)
        sigma_tt_hat = sigma_tt_hat.reshape(-1)
        return mu_s, mu_t, sigma_ss, sigma_st, sigma_tt_hat

    def _calculate_posterior_predictive_from_joint_distribution(self, xt, xs, pt1s1, version='numpy'):
        if version == 'pytorch':
            pt = self.predict_proba_torch(xt)
        else:
            pt = self.predict_proba(xt)

        assert (pt.shape == (1, 2))
        pt0, pt1 = pt[0, 0], pt[0, 1]
        ps = self.predict_proba(xs)
        ps0, ps1 = ps[:, 0], ps[:, 1]
        if version == 'pytorch':
            ps1 = torch.tensor(ps1)
        pt0s1 = ps1 - pt1s1
        ps1_t1 = pt1s1 / pt1
        ps0_t1 = 1 - ps1_t1
        ps1_t0 = pt0s1 / pt0
        ps0_t0 = 1 - ps1_t0

        if version == 'pytorch':
            column_stack = torch.column_stack
        else:
            column_stack = np.column_stack
        ps_t0 = column_stack((ps0_t0, ps1_t0))
        ps_t1 = column_stack((ps0_t1, ps1_t1))

        return ps_t0, ps_t1

    def OneStepPredict(self, xt, xs, version='numpy'):
        if version == 'pytorch':
            calculate_mean_variance = self._calculate_mean_and_variance_torch
            erf = torch.erf
            sqrt = torch.sqrt
            zeros = torch.zeros
        else:
            calculate_mean_variance = self._calculate_mean_and_variance
            erf = special.erf
            sqrt = np.sqrt
            zeros = np.zeros

        mu_s, mu_t, sigma_ss, sigma_st, sigma_tt_hat = calculate_mean_variance(xt, xs)
        sigma_s = np.sqrt(sigma_ss)
        Phi = lambda x: 0.5 * (erf(x / math.sqrt(2)) + 1)

        def func4(f0):
            # This function use hermite Gaussian quadrature,
            # return: a x_num array of value with index corresponding to xs.
            # term3 = 1/(sigma1*math.sqrt(2*math.pi))*math.exp(-0.5*(x3)**2) is normalized as Gaussian function

            fs = f0 * math.sqrt(2) * sigma_s + mu_s
            mu_t_hat = mu_t + sigma_st / sigma_ss * (fs - mu_s)
            ft_hat = mu_t_hat / sqrt(sigma_tt_hat + 1)
            term1 = Phi(ft_hat)
            term2 = Phi(fs)
            return term1 * term2 / math.sqrt(math.pi)  # math.sqrt(math.pi) is the constant for normalized Gaussian

        # joint distribution
        pt1s1 = zeros(len(xs))
        for i, f0 in enumerate(A[0]):
            pt1s1 += func4(f0) * A[1][i]

        ps_t0, ps_t1 = self._calculate_posterior_predictive_from_joint_distribution(xt, xs, pt1s1, version=version)
        return ps_t0, ps_t1

    def DataApprox(self, x):
        anum = 3
        X = self.gpc.X
        Y = self.gpc.Y
        kernel = self.parameters[2]
        d = kernel.lengthscale
        l = anum * d
        xlower = np.maximum(self.xinterval[0], x - l)
        xupper = np.minimum(self.xinterval[1], x + l)
        # idxarray = np.all(X >= xlower, axis=1) & np.all(X <= xupper, axis=1)

        X, idxarray = util.Xtruncated(xlower, xupper, X, self.f_num)

        Y = Y[idxarray].reshape(-1, 1)

        return X, Y

    def UpdateNew(self, x, y):  ##############################################################
        x = x.reshape(-1, self.f_num)
        X = np.concatenate((self.gpc.X, x), axis=0)
        Y = np.concatenate((self.gpc.Y, [[y]]), axis=0)
        # parameters = self.parameters
        # parameters[2] = self.gpc.kern
        # model2 = Model(X, Y, parameters, optimize = False)
        model2 = self.ModelTrain(X, Y)
        return model2

    def ModelTrain(self, X, Y):
        parameters = self.parameters
        parameters[2] = self.gpc.kern
        model2 = Model(X, Y, parameters, optimize=False)
        return model2

    def Update(self, x, y, optimize=False):  ########################################################################
        x = x.reshape(-1, self.f_num)
        X = np.concatenate((self.gpc.X, x))
        Y = np.concatenate((self.gpc.Y, [[y]]), axis=0)

        # lik = self.parameters[3]
        m = GPy.core.GP(X=X,
                        Y=Y,
                        kernel=self.gpc.kern,
                        inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                        likelihood=self.lik)
        if optimize:
            m.optimize_restarts(optimizer='bfgs', num_restarts=40, max_iters=200, verbose=False)
        self.gpc = m
        self.check_hyper_boundary()

    def XspaceGenerate(self, x_num, xspace_all):
        if xspace_all is not None:
            if x_num >= len(xspace_all):
                return xspace_all
            sampleidx = choice(range(xspace_all.shape[0]), x_num, replace=False)  #############################
            return xspace_all[sampleidx]
        xspace = np.random.uniform(self.xinterval[0], self.xinterval[1], (x_num, self.f_num))
        return xspace

    def ObcClassifierError(self, x_num, data):
        xspace = self.XspaceGenerate(x_num, data.xspace())
        pyTheta = self.predict_proba(xspace)
        yhat = np.argmax(pyTheta, axis=1)
        truth = GroundTruthFunction(xspace, data)
        classifier_error = (abs(yhat - truth) > 0.01).astype(float)
        return classifier_error.mean()



    def XspaceGenerateApprox(self, x_num, x):

        d = self.gpc.kern.lengthscale.item()
        if self.f_num > 1:
            cov = np.eye(self.f_num) * d
            mean = x.reshape(-1)
            xspace = np.random.multivariate_normal(mean=mean, cov=cov, size=x_num)
        else:
            xspace = np.random.normal(x, d, (x_num, 1))  ##this is only for single dimension
        # idxarray = np.all(xspace >= xinterval[0], axis=1) & np.all(xspace <= xinterval[1], axis = 1)
        # xspace = xspace[idxarray].reshape(-1, f_num)
        xspace = norm.rvs(size=(x_num, self.f_num), loc=x, scale=d)
        xspace, _ = util.Xtruncated(self.xinterval[0], self.xinterval[1], xspace, self.f_num)
        wspace_log_array = norm.logpdf(xspace, loc=x, scale=d)
        wspace_log = np.sum(wspace_log_array, axis=1) + np.log(x_num / len(xspace))
        _, px_log = util.setting_px(self.f_num, self.xinterval)

        return xspace, wspace_log, px_log



class Model_LS():

    def __init__(self, X, Y, parameters=None,
                 optimize=False, xinterval=(0., 1.)):  #####################################################

        self.parameters = parameters
        self.f_num = parameters[0]
        self.c_num = parameters[1]
        self.kernel = parameters[2]
        self.lik = parameters[3]
        self.m = LabelSpreading()
        self.m.fit(X, Y.ravel())
        self.X = X
        self.Y = Y


    def predict_proba(self, x):
        x = x.reshape(-1, self.f_num)
        M = len(x) // 1000
        if M <= 1:
            py_1 = self.m.predict_proba(x)[:, 1:2]
            py_0 = 1 - py_1
            pymat = np.concatenate((py_0, py_1), axis=1)  # pymat size x_num*cnum
            return pymat
        else:
            pymat1 = np.zeros((len(x), 2))
            for m in range(M):
                idx = range(m * 1000, m * 1000 + 1000)
                pymat1[idx, 1:2] = self.m.predict_proba(x[idx, :])[:, 1:2]
                pymat1[idx, 0] = 1 - pymat1[idx, 1]
            idx = range(m * 1000 + 1000, len(x))
            pymat1[idx, 1:2] = self.m.predict_proba(x[idx, :])[:, 1:2]
            pymat1[idx, 0] = 1 - pymat1[idx, 1]

            return pymat1

    def Update(self, x, y, optimize=False):  ########################################################################
        x = x.reshape(-1, self.f_num)
        self.X = np.concatenate((self.X, x))
        self.Y = np.concatenate((self.Y, [[y]]), axis=0)

        # lik = self.parameters[3]
        self.m = LabelSpreading()
        self.m.fit(self.X, self.Y.ravel())

    def XspaceGenerate(self, x_num, xspace_all):
        if xspace_all is not None:
            if x_num >= len(xspace_all):
                return xspace_all
            sampleidx = choice(range(xspace_all.shape[0]), x_num, replace=False)  #############################
            return xspace_all[sampleidx]
        xspace = np.random.uniform(self.xinterval[0], self.xinterval[1], (x_num, self.f_num))
        return xspace

    def ObcClassifierError(self, x_num, data):
        xspace = self.XspaceGenerate(x_num, data.xspace())
        pyTheta = self.predict_proba(xspace)
        yhat = np.argmax(pyTheta, axis=1)
        truth = GroundTruthFunction(xspace, data)
        classifier_error = (abs(yhat - truth) > 0.01).astype(float)
        return classifier_error.mean()


class Model_LP():

    def __init__(self, X, Y, parameters=None,
                 optimize=False, xinterval=(0., 1.)):  #####################################################

        self.parameters = parameters
        self.f_num = parameters[0]
        self.c_num = parameters[1]
        self.kernel = parameters[2]
        self.lik = parameters[3]
        self.m = LabelPropagation()
        self.m.fit(X, Y.ravel())
        self.X = X
        self.Y = Y


    def predict_proba(self, x):
        x = x.reshape(-1, self.f_num)
        M = len(x) // 1000
        if M <= 1:
            py_1 = self.m.predict_proba(x)[:, 1:2]
            py_0 = 1 - py_1
            pymat = np.concatenate((py_0, py_1), axis=1)  # pymat size x_num*cnum
            return pymat
        else:
            pymat1 = np.zeros((len(x), 2))
            for m in range(M):
                idx = range(m * 1000, m * 1000 + 1000)
                aaa = self.m.predict_proba(x[idx, :])
                pymat1[idx, 1:2] = self.m.predict_proba(x[idx, :])[:, 1:2]
                pymat1[idx, 0] = 1 - pymat1[idx, 1]
            idx = range(m * 1000 + 1000, len(x))
            pymat1[idx, 1:2] = self.m.predict_proba(x[idx, :])[:, 1:2]
            pymat1[idx, 0] = 1 - pymat1[idx, 1]

            return pymat1


    def Update(self, x, y, optimize=False):  ########################################################################
        x = x.reshape(-1, self.f_num)
        self.X = np.concatenate((self.X, x))
        self.Y = np.concatenate((self.Y, [[y]]), axis=0)

        # lik = self.parameters[3]
        self.m = LabelPropagation()
        self.m.fit(self.X, self.Y.ravel())

    def XspaceGenerate(self, x_num, xspace_all):
        if xspace_all is not None:
            if x_num >= len(xspace_all):
                return xspace_all
            sampleidx = choice(range(xspace_all.shape[0]), x_num, replace=False)  #############################
            return xspace_all[sampleidx]
        xspace = np.random.uniform(self.xinterval[0], self.xinterval[1], (x_num, self.f_num))
        return xspace

    def ObcClassifierError(self, x_num, data):
        xspace = self.XspaceGenerate(x_num, data.xspace())
        pyTheta = self.predict_proba(xspace)
        yhat = np.argmax(pyTheta, axis=1)
        truth = GroundTruthFunction(xspace, data)
        classifier_error = (abs(yhat - truth) > 0.01).astype(float)
        return classifier_error.mean()


class Model_regression():

    def __init__(self, X, Y, parameters=None,
                 optimize=False, xinterval=(0., 1.), hyper_inteval=None):  #####################################################
        # __slots__ = ['parameters', 'f_num', 'gpc', 'xinterval', 'optimize', 'lik', 'c_num']
        # X size is x_num*fnum
        # Y size is x_num
        if (parameters is None) or (xinterval is None):
            raise('ERROR!!!!!!!!!!!!!!!!!')
        self.parameters = parameters
        self.f_num = parameters[0]
        self.c_num = parameters[1]
        self.kernel = parameters[2]
        self.lik = parameters[3]
        self.hyper_inteval = hyper_inteval
        mf = GPy.core.Mapping(self.f_num, 1)
        mf.f = lambda x: Y.mean()
        mf.update_gradients = lambda a, b: 0
        mf.gradients_X = lambda a, b: 0
        m = GPy.core.GP(X=X,
                        Y=Y,
                        mean_function=mf,
                        kernel=self.kernel,
                        inference_method=GPy.inference.latent_function_inference.exact_gaussian_inference.ExactGaussianInference(),
                        likelihood=self.lik)
        if optimize:
            m.optimize_restarts(optimizer='bfgs', num_restarts=40, max_iters=2000, verbose=False)
        self.gpc = m
        self.check_hyper_boundary()
        self.xinterval = xinterval
        self.optimize = optimize

    def check_hyper_boundary(self):
        if self.hyper_inteval is None:
            raise('!!!!!!!!')

        mf = GPy.core.Mapping(self.f_num, 1)
        mf.f = lambda x: self.gpc.Y.mean()
        mf.update_gradients = lambda a, b: 0
        mf.gradients_X = lambda a, b: 0
        if self.gpc.kern.lengthscale.item() < self.hyper_inteval[0][0]:
            self.kernel = GPy.kern.RBF(self.f_num, variance=self.gpc.kern.variance.item(), lengthscale=self.hyper_inteval[0][0])
            self.gpc = GPy.core.GP(X=self.gpc.X,
                            Y=self.gpc.Y,
                            mean_function=mf,
                            kernel=self.kernel,
                            inference_method=GPy.inference.latent_function_inference.exact_gaussian_inference.ExactGaussianInference(),
                            likelihood=self.lik)
        elif self.gpc.kern.lengthscale.item() > self.hyper_inteval[0][1]:
            self.kernel = GPy.kern.RBF(self.f_num, variance=self.gpc.kern.variance.item(),
                                  lengthscale=self.hyper_inteval[0][1])
            self.gpc = GPy.core.GP(X=self.gpc.X,
                                   Y=self.gpc.Y,
                                   mean_function=mf,
                                   kernel=self.kernel,
                                   inference_method=GPy.inference.latent_function_inference.exact_gaussian_inference.ExactGaussianInference(),
                                   likelihood=self.lik)

        if self.gpc.kern.variance.item() < self.hyper_inteval[1][0]:
            self.kernel = GPy.kern.RBF(self.f_num, variance=self.hyper_inteval[1][0],
                                       lengthscale=self.gpc.kern.lengthscale.item())
            self.gpc = GPy.core.GP(X=self.gpc.X,
                                   Y=self.gpc.Y,
                                   mean_function=mf,
                                   kernel=self.kernel,
                                   inference_method=GPy.inference.latent_function_inference.exact_gaussian_inference.ExactGaussianInference(),
                                   likelihood=self.lik)
        elif self.gpc.kern.variance.item() > self.hyper_inteval[1][1]:
            self.kernel = GPy.kern.RBF(self.f_num, variance=self.hyper_inteval[1][1],
                                       lengthscale=self.gpc.kern.lengthscale.item())
            self.gpc = GPy.core.GP(X=self.gpc.X,
                                   Y=self.gpc.Y,
                                   mean_function=mf,
                                   kernel=self.kernel,
                                   inference_method=GPy.inference.latent_function_inference.exact_gaussian_inference.ExactGaussianInference(),
                                   likelihood=self.lik)
    def predict_proba(self, x):
        x = x.reshape(-1, self.f_num)
        M = len(x) // 1000
        if M <= 1:
            pymat = self.gpc.predict(x)
            # py_0 = 1 - py_1
            # pymat = np.concatenate((py_0, py_1), axis=1)  # pymat size x_num*cnum
            return np.concatenate(pymat, axis=1)
        else:
            # raise('!!!!!!!!!')
            pymat1 = np.zeros((len(x), 2))
            for m in range(M):
                idx = range(m * 1000, m * 1000 + 1000)
                pymat1[idx, :] = np.concatenate(self.gpc.predict(x[idx, :]), axis=1)
            idx = range(m * 1000 + 1000, len(x))
            pymat1[idx, :] = np.concatenate(self.gpc.predict(x[idx, :]), axis=1)

            return pymat1

    def _noiseless_predict_torch(self, xt):
        woodbury_inv = torch.tensor(self.gpc.posterior.woodbury_inv)
        woodbury_vector = torch.tensor(self.gpc.posterior.woodbury_vector)
        X_ = torch.tensor(self.gpc.X)
        K = self.K

        mu_t = K(xt, X_) @ woodbury_vector
        sigma_tt = K(xt, xt) - K(xt, X_) @ woodbury_inv @ K(X_, xt)

        return mu_t, sigma_tt

    def K(self, xt, xs):

        kern = self.gpc.kern
        assert (kern.name == 'rbf')  # the function is only coded for rbf kernel

        l1 = kern.lengthscale.item()
        l2 = kern.variance.item()
        Kts = l2 * torch.exp(-torch.cdist(xt, xs) ** 2 / 2 / l1 ** 2)
        return Kts

    def predict_proba_torch(self, xt):
        xt = xt.reshape(-1, self.f_num)
        assert (type(xt) == torch.Tensor)

        Phi = lambda x: 0.5 * (torch.erf(x / math.sqrt(2)) + 1)
        mu_t, sigma_tt = self._noiseless_predict_torch(xt)

        ft_hat = mu_t / torch.sqrt(sigma_tt + 1)
        py_1 = Phi(ft_hat)
        py_0 = 1 - py_1
        pymat = torch.cat([py_0, py_1], axis=1)
        return pymat

    def _calculate_mean_and_variance(self, xt, xs):
        xt = xt.reshape(-1, self.f_num)
        xs = xs.reshape(-1, self.f_num)
        muvar = self.gpc.predict_noiseless(np.concatenate((xs, xt)), full_cov=False)
        mu = muvar[0].reshape(-1)
        mu_s = mu[0:-1]
        mu_t = mu[-1]

        var = muvar[1].reshape(-1)
        sigma_ss = var[0:-1]
        X_ = self.gpc.X
        sigma_st = self.gpc.kern.K(xs, xt) - self.gpc.kern.K(xs,
                                                             X_) @ self.gpc.posterior.woodbury_inv @ self.gpc.kern.K(X_,
                                                                                                                     xt)
        sigma_st = sigma_st.reshape(-1)
        sigma_tt = var[-1]
        sigma_tt_hat = sigma_tt - sigma_st ** 2 / sigma_ss
        return mu_s, mu_t, sigma_ss, sigma_st, sigma_tt_hat

    def _calculate_mean_and_variance_torch(self, x1, x2):
        xt = x1.reshape(-1, self.f_num)
        xs = torch.tensor(x2.reshape(-1, self.f_num))

        X_ = torch.tensor(self.gpc.X)
        woodbury_inv = torch.tensor(self.gpc.posterior.woodbury_inv)
        woodbury_vector = torch.tensor(self.gpc.posterior.woodbury_vector)

        muvar = self.gpc.predict_noiseless(x2, full_cov=False)
        mu_s = torch.tensor(muvar[0])
        sigma_ss = torch.tensor(muvar[1])

        K = self.K
        mu_t, sigma_tt = self._noiseless_predict_torch(xt)
        sigma_st = K(xs, xt) - K(xs, X_) @ woodbury_inv @ K(X_, xt)
        sigma_tt_hat = sigma_tt - sigma_st ** 2 / sigma_ss
        mu_s = mu_s.reshape(-1)
        mu_t = mu_t.reshape(-1)
        sigma_ss = sigma_ss.reshape(-1)
        sigma_st = sigma_st.reshape(-1)
        sigma_tt_hat = sigma_tt_hat.reshape(-1)
        return mu_s, mu_t, sigma_ss, sigma_st, sigma_tt_hat

    def _calculate_posterior_predictive_from_joint_distribution(self, xt, xs, pt1s1, version='numpy'):
        if version == 'pytorch':
            pt = self.predict_proba_torch(xt)
        else:
            pt = self.predict_proba(xt)

        assert (pt.shape == (1, 2))
        pt0, pt1 = pt[0, 0], pt[0, 1]
        ps = self.predict_proba(xs)
        ps0, ps1 = ps[:, 0], ps[:, 1]
        if version == 'pytorch':
            ps1 = torch.tensor(ps1)
        pt0s1 = ps1 - pt1s1
        ps1_t1 = pt1s1 / pt1
        ps0_t1 = 1 - ps1_t1
        ps1_t0 = pt0s1 / pt0
        ps0_t0 = 1 - ps1_t0

        if version == 'pytorch':
            column_stack = torch.column_stack
        else:
            column_stack = np.column_stack
        ps_t0 = column_stack((ps0_t0, ps1_t0))
        ps_t1 = column_stack((ps0_t1, ps1_t1))

        return ps_t0, ps_t1

    def OneStepPredict(self, xt, xs, version='numpy'):
        if version == 'pytorch':
            calculate_mean_variance = self._calculate_mean_and_variance_torch
            erf = torch.erf
            sqrt = torch.sqrt
            zeros = torch.zeros
        else:
            calculate_mean_variance = self._calculate_mean_and_variance
            erf = special.erf
            sqrt = np.sqrt
            zeros = np.zeros

        mu_s, mu_t, sigma_ss, sigma_st, sigma_tt_hat = calculate_mean_variance(xt, xs)
        sigma_s = np.sqrt(sigma_ss)
        Phi = lambda x: 0.5 * (erf(x / math.sqrt(2)) + 1)

        def func4(f0):
            # This function use hermite Gaussian quadrature,
            # return: a x_num array of value with index corresponding to xs.
            # term3 = 1/(sigma1*math.sqrt(2*math.pi))*math.exp(-0.5*(x3)**2) is normalized as Gaussian function

            fs = f0 * math.sqrt(2) * sigma_s + mu_s
            mu_t_hat = mu_t + sigma_st / sigma_ss * (fs - mu_s)
            ft_hat = mu_t_hat / sqrt(sigma_tt_hat + 1)
            term1 = Phi(ft_hat)
            term2 = Phi(fs)
            return term1 * term2 / math.sqrt(math.pi)  # math.sqrt(math.pi) is the constant for normalized Gaussian

        # joint distribution
        pt1s1 = zeros(len(xs))
        for i, f0 in enumerate(A[0]):
            pt1s1 += func4(f0) * A[1][i]

        ps_t0, ps_t1 = self._calculate_posterior_predictive_from_joint_distribution(xt, xs, pt1s1, version=version)
        return ps_t0, ps_t1

    def DataApprox(self, x):
        anum = 3
        X = self.gpc.X
        Y = self.gpc.Y
        kernel = self.parameters[2]
        d = kernel.lengthscale
        l = anum * d
        xlower = np.maximum(self.xinterval[0], x - l)
        xupper = np.minimum(self.xinterval[1], x + l)
        # idxarray = np.all(X >= xlower, axis=1) & np.all(X <= xupper, axis=1)

        X, idxarray = util.Xtruncated(xlower, xupper, X, self.f_num)

        Y = Y[idxarray].reshape(-1, 1)

        return X, Y

    def UpdateNew(self, x, y):  ##############################################################
        x = x.reshape(-1, self.f_num)
        X = np.concatenate((self.gpc.X, x), axis=0)
        Y = np.concatenate((self.gpc.Y, [[y]]), axis=0)
        # parameters = self.parameters
        # parameters[2] = self.gpc.kern
        # model2 = Model(X, Y, parameters, optimize = False)
        model2 = self.ModelTrain(X, Y)
        return model2

    def ModelTrain(self, X, Y):
        parameters = self.parameters
        parameters[2] = self.gpc.kern
        model2 = Model(X, Y, parameters, optimize=False)
        return model2

    def Update(self, x, y, optimize=False):  ########################################################################
        x = x.reshape(-1, self.f_num)
        X = np.concatenate((self.gpc.X, x))
        Y = np.concatenate((self.gpc.Y, [[y]]), axis=0)

        # lik = self.parameters[3]
        mf = GPy.core.Mapping(self.f_num, 1)
        mf.f = lambda x: Y.mean()
        mf.update_gradients = lambda a, b: 0
        mf.gradients_X = lambda a, b: 0
        m = GPy.core.GP(X=X,
                        Y=Y,
                        mean_function=mf,
                        kernel=self.gpc.kern,
                        likelihood=self.lik)
        if optimize:
            m.optimize_restarts(optimizer='bfgs', num_restarts=40, max_iters=200, verbose=False)
        self.gpc = m
        self.check_hyper_boundary()

    def XspaceGenerate(self, x_num, xspace_all):
        if xspace_all is not None:
            if x_num >= len(xspace_all):
                return xspace_all
            sampleidx = choice(range(xspace_all.shape[0]), x_num, replace=False)  #############################
            return xspace_all[sampleidx]
        xspace = np.random.uniform(self.xinterval[0], self.xinterval[1], (x_num, self.f_num))
        return xspace

    def ObcClassifierError(self, x_num, data, threshold):
        xspace = self.XspaceGenerate(x_num, data.xspace())
        pyTheta = self.predict_proba(xspace)[:, 0]
        yhat = (pyTheta > threshold).astype(float)
        truth = (GroundTruthFunction(xspace, data) > threshold).astype(float)
        classifier_error = (abs(yhat - truth) > 0.01).astype(float)
        return classifier_error.mean()



    def XspaceGenerateApprox(self, x_num, x):

        d = self.gpc.kern.lengthscale.item()
        if self.f_num > 1:
            cov = np.eye(self.f_num) * d
            mean = x.reshape(-1)
            xspace = np.random.multivariate_normal(mean=mean, cov=cov, size=x_num)
        else:
            xspace = np.random.normal(x, d, (x_num, 1))  ##this is only for single dimension
        # idxarray = np.all(xspace >= xinterval[0], axis=1) & np.all(xspace <= xinterval[1], axis = 1)
        # xspace = xspace[idxarray].reshape(-1, f_num)
        xspace = norm.rvs(size=(x_num, self.f_num), loc=x, scale=d)
        xspace, _ = util.Xtruncated(self.xinterval[0], self.xinterval[1], xspace, self.f_num)
        wspace_log_array = norm.logpdf(xspace, loc=x, scale=d)
        wspace_log = np.sum(wspace_log_array, axis=1) + np.log(x_num / len(xspace))
        _, px_log = util.setting_px(self.f_num, self.xinterval)

        return xspace, wspace_log, px_log

