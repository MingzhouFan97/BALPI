from dataset import DataSet_from_csv
import numpy as np
import experiment
import optimization
import utilityfunction
import os
import matplotlib

if __name__ == '__main__':
    init = 20   #The number of initial samples

    iters = 80  #Choose the iterations you need to run

    for k in range(10):
        hyper_int = [[0.05, .15], [1e-4, 2.]]   #hyperparameters bounds
        save_dir = f'results_BCC_{init}/' #diretory for saving
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        data = DataSet_from_csv('./data/toy_data_regression.csv')#Find the dataset
        data.generate_plt().show()
        data.set_x_int((0., 1.))
        optimizer = lambda util, model, data: optimization.MCSelector(util, model, 1000, data)
        U_UCB = lambda x, model: utilityfunction.U_UCBS(x, model, .8, .5)
        #3 is the dimension of the input, 2 is the number of classes
        exp = experiment.Experiement_fixed_discrete_data_regression(data, 3, 2, hyper_int, optimizer, U_UCB,
                                                                    threshold=0.8)
        exp.initialize(init)
        for i in range(iters):
            print(save_dir, i, k)
            exp.perform_one_iteration(retrain=True)
            exp.generate_figure(save_dir, f'{i}', threshold=.8)
        np.savetxt(save_dir + f'error{k}.csv', np.array(exp.error_recording()), delimiter=',')