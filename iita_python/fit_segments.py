import numpy as np
import numpy.typing as npt
import pandas as pd
from .dataset import Dataset
from .quasiorder import QuasiOrder

def orig_error_rate(data: Dataset, qo: QuasiOrder) -> tuple[npt.NDArray, float]:
    """
    Calculates the error rate for a given dataset and quasiorder according to the original IITA algorithm\n
    """
    qo_edges = qo.get_edge_list()
    p = data.rp.to_numpy().sum(axis=0) / data.subjects

    error = 0
    for a, b in qo_edges:
        error += data.ce.iloc[a, b] / (p[b] * data.subjects)
    
    error /= len(qo_edges)

    return p, error

def mini_error_rate(data: Dataset, qo: QuasiOrder) -> tuple[npt.NDArray, float]:
    """
    Calculates the error rate for a given dataset and quasiorder according to the minimized IITA algorithm\n
    """
    qo_edges = qo.get_edge_list()
    p = data.rp.to_numpy().sum(axis=0)

    x = [0, 0, 0, 0]
    for a in range(data.items):
        for b in range(data.items):
            if (a == b): continue

            if (qo.full_matrix[a][b]):
                x[1] += -2 * data.ce.iloc[a, b] * p[b]
                x[3] += 2 * p[b] ** 2
            elif (qo.full_matrix[b][a]):
                x[0] += -2 * data.ce.iloc[a, b] * p[a] + 2 * p[a] * p[b] - 2 * p[a] ** 2
                x[2] += 2 * p[a] ** 2

    error = - (x[0] + x[1]) / (x[2] + x[3])

    return p / data.subjects, error

def orig_expected_ce(data: Dataset, qo: QuasiOrder, p, error) -> npt.NDArray:
    """
    Calculates the expected counterexamples matrix for a given dataset and quasiorder according to the original IITA algorithm\n
    """

    expected_ce = np.zeros(data.ce.shape)

    for i in range(data.items):
        for j in range(data.items):
            if (i == j): continue

            if (qo.full_matrix[i][j]):
                expected_ce[i][j] = error * p[j] * data.subjects
            else:
                expected_ce[i][j] = (1 - p[i]) * p[j] * data.subjects * (1 - error)
    
    return expected_ce

def corr_expected_ce(data: Dataset, qo: QuasiOrder, p, error) -> npt.NDArray:
    """
    Calculates the expected counterexamples matrix for a given dataset and quasiorder according to the corrected IITA algorithm\n
    """

    expected_ce = np.zeros(data.ce.shape)

    for i in range(data.items):
        for j in range(data.items):
            if (i == j): continue

            if (qo.full_matrix[i][j]):
                expected_ce[i][j] = error * p[j] * data.subjects
            elif (not qo.full_matrix[j][i]):
                expected_ce[i][j] = (1 - p[i]) * p[j] * data.subjects
            else:
                expected_ce[i][j] = (p[j] * data.subjects) - ((p[i] - p[i] * error) * data.subjects)
    
    return expected_ce

def avg_squared_diff(a: npt.NDArray, b: npt.NDArray) -> float:
    """
    Calculates the average of squared differences between two arrays\n
    """
    return ((np.array(a).flatten() - np.array(b).flatten()) ** 2).sum() / (a.shape[0] * (a.shape[0] - 1))