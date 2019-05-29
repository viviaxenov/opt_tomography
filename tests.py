import os
import time
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from skimage import io

from mirror_descent import MirrorDescent
from gradient_descent import GradientDescent
from dual_averaging import DualAveragingRPP
from composite_mirror_prox import CompositeMirrorProx
from utils import create_noisy_image


PROF_DATA = {}


def with_time_elapsed(function):
    """
    a method-friendly wrapper to estimate elapsed time
    for each function modified by it
    """

    @wraps(function)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        function(self, *args, **kwargs)
        elapsed_time = time.time() - start_time

        if function.__name__ not in PROF_DATA:
            PROF_DATA[function.__name__] = elapsed_time

    return wrapper


class MirrorDescentTest(MirrorDescent):

    def __init__(self, x_0, f, sg, eta, simplex_size):
        super(MirrorDescentTest, self).__init__(
            x_0, f, sg, eta, simplex_size)
        self.name = "Mirror Descent"
        self.x_0 = x_0
        self.iteration_steps = np.array([x_0])


class GradientDescentTest(GradientDescent):

    def __init__(self, x_0, f, sg, eta, simplex_size):
        super(GradientDescentTest, self).__init__(
            x_0, sg, eta, simplex_size
        )
        self.name = "Gradient Descent"
        self.x_0 = x_0
        self.iteration_steps = np.array([x_0])


class DualAveragingTest (DualAveragingRPP):

    def __init__(self, x_0, f, sg, eta, simplex_size):
        super(DualAveragingTest, self).__init__(
            x_0, f, sg, eta, simplex_size)
        self.name = "Dual Averaging"
        self.x_0 = x_0
        self.iteration_steps = np.array([x_0])


class CompositeMirrorProxTest(CompositeMirrorProx):
    def __init__(self, x_0, f, sg, a, w, alpha, simplex_size):
        super(CompositeMirrorProxTest, self).__init__(
                x_0, a, w, alpha, simplex_size
        )
        self.name = "Composite Mirror Prox."
        self.x_0 = x_0
        self.iteration_steps = np.array([x_0])


class Tests:
    """
    A class for numerical experiments to estimate the efficiency of different algorithms
    on practice
    """
    def __init__(self, _target_function, _target_grad, source_image, stoch_matrix):

        self.A = load_npz(
            os.path.join(os.getcwd(), stoch_matrix)
        ).tocsr()
        filename = os.path.join(os.getcwd(), source_image)
        try:
            self.img_shape, self.true_image, self.w = create_noisy_image(filename, self.A)
        except FileNotFoundError as err:
            print(err.args)

        self.simplex_size = np.linalg.norm(self.w, 1)  # size of initial simplex
        # initial point is the mass center of simplex
        self.x_0 = np.full_like(self.true_image, 1 / self.true_image.shape[0]) * self.simplex_size
        self.method_list = []

        self.f = lambda x: _target_function(x, self.w)
        self.grad = lambda x: _target_grad(x, self.w)

    def add_method(self, method):
        self.method_list.append(method)

    @with_time_elapsed
    def run_fixed_iter_amount(self, method, iter_amount=1000):
        method.iteration_steps = np.array([method.x_0])
        for i in range(iter_amount):
            method.iteration_steps = np.append(
                method.iteration_steps,
                np.array([method.iteration()]),
                axis=0
            )

    @with_time_elapsed
    def run_fixed_eps(self, method, eps: float):
        """
        Iterations continue until the required precision eps
        """
        method.iteration_steps = np.array([method.x_0])
        while abs(self.f(self.true_image) - self.f(method.iteration_steps[-1])) > eps:
            np.append(
                method.iteration_steps,
                np.array([method.iteration()]),
                axis=0
            )

    def get_error_evolution(self, method):
        """
        :param method
        :return: np.array of relative objective function error evolution during the iteration process
        """

        true_objective_value = self.f(self.true_image)
        objective_rel_err = np.array([
            abs(self.f(method.iteration_steps[i]) - true_objective_value) /
            abs(true_objective_value)
            for i in range(method.iteration_steps.shape[0])]
        )

        return objective_rel_err

    def test_equal_iteration_amount(self, iterations: int):
        """
        :param iterations: the number of iterations
        :return: np.array of relative objective function error evolution
        during the iteration process
        """

        err_evolution = []

        for m in self.method_list:
            self.run_fixed_iter_amount(m, iterations)
            err_evolution.append(self.get_error_evolution(m))

        return err_evolution

    def test_equal_precision(self, eps: float):
        """
        :param eps: required precision
        :return: np.array of relative objective function error evolution
        during the iteration process
        """

        err_evolution = []

        for m in self.method_list:
            self.run_fixed_eps(m, eps)
            err_evolution.append(self.get_error_evolution(m))

        return err_evolution

    def run(self, iter_amount):

        err_evo = self.test_equal_iteration_amount(iter_amount)

        for i in range(len(self.method_list)):
            plt.plot(np.arange(iter_amount + 1), err_evo[i], label=self.method_list[i].name)
            img = self.method_list[i].iteration_steps[-1]
            img = np.array([i if i <= 255 else 255 for i in np.nditer(img)]).reshape(self.img_shape)
            io.imsave(f"{self.method_list[i].name}.jpg", img.astype(np.uint8))

        plt.title(f"{iter_amount} iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Objective relative error")
        plt.legend()
        plt.savefig('fixed_iterations_amount.png')










