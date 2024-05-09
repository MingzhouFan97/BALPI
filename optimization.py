'''
Optimize method for continuous search space
include:
1. Exhausive search
2. Non-gradient method
3. Gradient ascent
'''
import numpy as np
import torch


def MCSelector(func, model, mc_search_num=1000, data=None):
    xspace = model.XspaceGenerate(mc_search_num, data.xspace())

    utilitymat = np.zeros(mc_search_num) + float('-Inf')

    if hasattr(model, 'multi_hyper') and model.multi_hyper:
        for i, x in enumerate(xspace):
            if hasattr(model, 'is_real_data') and model.is_real_data:
                if i in model.dataidx:
                    continue
            x = xspace[i:i + 1]
            for m in model.modelset:
                utilitymat[i] += func(x, m)
    else:
        for i, x in enumerate(xspace):
            if hasattr(model, 'is_real_data') and model.is_real_data:
                if i in model.dataidx:
                    continue
            x = xspace[i:i + 1]  # all the inputs should take 2d array
            utilitymat[i] = func(x, model)
    if np.isnan(utilitymat).any():
        pass
    utilitymat = np.nan_to_num(utilitymat, -100)
    max_value = np.max(utilitymat, axis=None)

    max_index = np.random.choice(np.flatnonzero(abs(utilitymat - max_value) < 1e-5))

    if hasattr(model, 'is_real_data') and model.is_real_data:
        model.dataidx = np.append(model.dataidx, max_index)

    x = xspace[max_index]

    return x, max_value


def RandomSampling(model):
    x = model.XspaceGenerate(1)
    max_value = 0
    return x, max_value


def SGD(func, model, mc_search_num=1000, learning_rate=0.001):
    # for mm in range(100):
    random_num = round(0.7 * mc_search_num)
    # x11, value11 = MCSelector(func, model, mc_search_num)
    x1, value1 = MCSelector(func, model, random_num)
    # x0 = model.XspaceGenerate(1).reshape(-1)
    x0 = torch.tensor(x1, requires_grad=True)
    optimizer = torch.optim.SGD([x0], lr=learning_rate)

    for _ in range(round(0.3 * mc_search_num)):
        loss = -func(x0, model, version='pytorch')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return x0.detach().numpy(), -loss
