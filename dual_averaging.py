import numpy as np
import scipy as sp
import scipy.optimize, scipy.stats

class DualAveragingMethod:
    def __init__(self, x_0: np.ndarray, f, sg, eta: np.float64, bounds: sp.optimize.Bounds, mirror_map='Rn'):
        """"
        Class for optimizing function using dual averaging method
        Args:
            x_0 - initial guess
            f - function to be optimized
            sg - function x: np.ndarray -> sg(x): np.ndarray returns a member of subgradient in point x
            eta - gradient descent parameter
            bounds - instance of scp.optimize.Bounds - defines the convex set X in which the function should be optimized
            mirror_map - string 'Rn' or 'R++' - defines mirror map function (see Bubeck, p.298 for definition)
        """
        self.f = f
        self.sg = sg
        self.eta = eta
        self.g = sg(x_0)
        self.x = x_0
        self.sum = x_0
        self.x_avg = x_0
        self.x_min = x_0
        self.bounds = bounds
        self.iterations = 0
        self.F = []
        self.jac = []

        if mirror_map == 'Rn':
            self.F = lambda x: 0.5*np.dot(x, x)
            self.jac = lambda x: self.eta*self.g + x
        elif mirror_map == 'R++':
            self.F = lambda x: np.log(x).T@x
            self.jac = lambda x: self.eta*self.g + np.log(x) + 1.
        else:
            raise ValueError('Unknown mirror map type. Can only be \'Rn\' or \'R++\'')

        self.step_function = lambda x: self.eta*np.dot(self.g, x) + self.F(x)

    def iteration(self):
        res: scipy.optimize.OptimizeResult = scipy.optimize.minimize(self.step_function, self.x, jac=self.jac,
                                                                     bounds=self.bounds, method='L-BFGS-B')
        self.iterations += 1
        if not res.success:
            raise RuntimeError(f'Failed to make iteration {self.iterations}. Cause of termination: {res.message}')

        if self.f(res.x) < self.f(self.x_min):
            self.x_min = res.x
        self.x = res.x
        self.sum += res.x
        self.x_avg = self.sum/self.iterations
        self.g += self.sg(self.x)




