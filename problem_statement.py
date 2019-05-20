import numpy as np
from scipy.sparse import csr


def target_function(x: np.array, a: csr, w: np.array):
    tmp = a.dot(x)
    return tmp - w * np.log(tmp)


def target_grad(x: np.array, a: csr, w: np.array):
    """
    :return: value of target function's gradient in point x
    """
    foo = a.transpose() @ np.ones(a.shape[1])
    tmp = w / (a @ x)
    bar = []
    for i in range(a.shape[1]):
        row_k = np.zeros(a.shape[1])
        row_k[i] = 1
        row_k = a @ row_k
        bar.append(row_k @ tmp)

    return foo - np.array(bar)


def eta_const(x: int):
    return 1e-2


def eta_fastest(x: int):
    raise NotImplementedError  # optimal step for fastest descent








