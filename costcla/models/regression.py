"""
This module include the cost-sensitive logistic regression method.
"""

# Authors: Alejandro Correa Bahnsen <al.bahnsen@gmail.com>
# License: BSD 3 clause

from ..optimization import GAcont
from ..metrics import cost_measure

import numpy as np
import math
from scipy.optimize import minimize

# TODO: Fix CSLog function
class CSLogisticRegression():
    def __init__(self):
        def sigmoid2(t):
            if t < -10:
                return 0.00000001
            else:
                return 1.0 / (1 + math.exp(-t))

        self.sigmoid = np.vectorize(sigmoid2)
        self.fitnessfunc = cost
        return None

    # TODO: Convert parameters to diccionary
    def fit(self, x, y, cost_mat, intercept=True, reg=0, method='BFGS', range1=None, params_ga=[100, 100, 10, 0.25]):
        #Function to fit a Logistic Regression
        #Input matrix X (not structure) and vector Y.
        #If jac!=False is because manual gradient is calculated
        #Otherwise gradient is calculated by the optimization function
        #On initial tests using the gradient make the algorithm run 2X faster

        setattr(self, 'intercept', intercept)
        if intercept == True:
            x = np.hstack((np.ones((x.shape[0], 1)), x))

        n = x.shape[1]
        initial_theta = np.zeros((n, 1))
        if method == "GA":
            if range1 == None:
                range1 = np.vstack((-10 * np.ones((1, n)), 10 * np.ones((1, n))))
            res = GAcont(n, self.fitnessfunc, params_ga[0], params_ga[1], CS=params_ga[2], MP=params_ga[3],
                         range=range1,
                         fargs=[y, x, cost_mat, reg])
            res.evaluate()
        else:
            res = minimize(self.fitnessfunc, initial_theta, (y, x, cost_mat, reg,), method=method,
                           options={'maxiter': 100, 'disp': True})

        setattr(self, 'theta', res.x)
        setattr(self, 'opti', res)
        setattr(self, 'hist', res.hist)
        setattr(self, 'full_hist', res.full_hist)

    def predict_proba(self, x_test):
        #Calculate the prediction of a LogRegression
        if self.intercept == True:
            x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))
        p = np.zeros((x_test.shape[0], 2))
        p[:, 1] = self.sigmoid(np.dot(x_test, self.theta))
        p[:, 0] = 1 - p[:, 1]
        return p

    def predict(self, x_test, cut_point=0.5):
        #Calculate the prediction of a LogRegression
        p = np.floor(self.predict_proba(x_test)[:, 1] + (1 - cut_point))
        return p.reshape(p.shape[0], )


import numpy as np, math, pp, thread

#global init Job_server_prof
def sigmoid(t):
    #missing to compare if this is faster than without vectorization, since it have to be calculated each time
    def sigmoid2(t):
        if t < -10:
            return 0.00000001
        else:
            return 1.0 / ( 1 + math.exp(-t))

    return np.vectorize(sigmoid2)(t)


Job_server_prof = pp.Server(restart=True)


def cost(theta, y, x, cost_mat, reg, parallel=12):
    #theta_i is a vector 1xCols where x.shape[1]=Cols
    #theta is a matrix NxCols
    #check if theta=array
    def cost_i(j, j1, theta1, y, x, cost_mat, reg=0.0):
        res = np.empty(((min(j * j1 + j1, theta1.shape[0]) - j * j1), 2))
        m = y.shape[0]
        for i in range(j * j1, min(j * j1 + j1, theta1.shape[0])):
            # res[i-j*j1,1]=cost_p((sigmoid(np.dot(x,theta1[i].transpose()))).reshape(y.shape),y,amt,ca)/m
            p = (sigmoid(np.dot(x, theta1[i].transpose()))).reshape(y.shape)
            res[i - j * j1, 1] = cost_measure(y, p, cost_mat)
            res[i - j * j1, 1] + reg / m * ((theta1[i] ** 2).sum())
            res[i - j * j1, 0] = i
        return res

    class Res:
        def __init__(self, N):
            self.res = np.zeros(N, dtype='<f8')
            self.lock = thread.allocate_lock()

        def save_res(self, value):
            self.lock.acquire()
            for i in range(value.shape[0]):
                self.res[int(value[i, 0])] = value[i, 1]
            self.lock.release()

    if theta.shape[0] == theta.size:
        theta = theta.reshape(1, theta.shape[0])
    N = theta.shape[0]
    res = Res(N)
    n_jobs = min(parallel, N)
    j1 = int(math.ceil(float(N) / n_jobs))
    for j in range(n_jobs):
        Job_server_prof.submit(cost_i, (j, j1, theta, y, x, cost_mat, 0.0), (sigmoid, cost_measure,),
                               ('numpy as np', 'math'), callback=res.save_res)
    Job_server_prof.wait()
    return res.res
