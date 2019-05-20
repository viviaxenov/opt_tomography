import numpy as np
import scipy as sp
import scipy.optimize, scipy.stats

class DualAveragingRPP:
    def __init__(self, x_0: np.ndarray, f, sg, eta: np.float64):
        """"
        Class for optimizing function using dual averaging method on R++ with mirror map \sum{x_i \log{x_i}}
        Args:
            x_0 - initial guess
            f - function to be optimized
            sg - function x: np.ndarray -> sg(x): np.ndarray returns a member of subgradient in point x
            eta - gradient descent parameter
        """
        self.f = f
        self.sg = sg
        self.eta = eta

        if(np.any(x_0 <= 0)):
            raise ValueError("Some of x_0 components are negative")

        self.g = sg(x_0)
        self.x = x_0
        self.sum = x_0
        self.x_avg = x_0
        self.x_min = x_0
        self.iterations = 0
        self.F = lambda x: x.T@np.log(x)


    def iteration(self):
        x_new = np.exp(-self.eta*self.g - np.log(self.x) - 1.)
        self.iterations += 1

        if self.f(x_new) < self.f(self.x_min):
            self.x_min = x_new
        self.x = x_new
        self.sum += x_new
        self.x_avg = self.sum/self.iterations
        self.g += self.sg(self.x)




