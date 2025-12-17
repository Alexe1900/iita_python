import numpy as np
import numpy.typing as npt
import pandas as pd
from .dataset import Dataset
from .quasiorder import QuasiOrder

def orig_iita_fit(data: Dataset, qo: QuasiOrder):
    """
    Calculates the original IITA fit metric for a given dataset and quasiorder\n
    """
    qo_edges = qo.get_edge_list()
    p = np.nansum(data.rp.to_numpy(), axis=0) / data.subjects

    error = 0
    for a, b in qo_edges:
        error += data.ce.iloc[a, b] / (p[b] * data.subjects)
    
    error /= len(qo_edges)

    expected_ce = np.zeros(data.ce.shape)

    for i in range(data.items):
        for j in range(data.items):
            if (i == j): continue

            if (qo.full_matrix[i][j]):
                expected_ce[i][j] = error * p[j] * data.subjects
            else:
                expected_ce[i][j] = (1 - p[i]) * p[j] * data.subjects * (1 - error)
    
    ce = data.ce.to_numpy().flatten()
    expected_ce = expected_ce.flatten()
    
    return ((ce - expected_ce) ** 2).sum() / (data.items**2 - data.items)

def corr_iita_fit(data: Dataset, qo: QuasiOrder):
    """
    Calculates the corrected IITA fit metric for a given dataset and quasiorder\n
    """
    qo_edges = qo.get_edge_list()
    p = np.nansum(data.rp.to_numpy(), axis=0) / data.subjects

    error = 0
    for a, b in qo_edges:
        error += data.ce.iloc[a, b] / (p[b] * data.subjects)
    
    error /= len(qo_edges)

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
    
    ce = data.ce.to_numpy().flatten()
    expected_ce = expected_ce.flatten()
    return ((ce - expected_ce) ** 2).sum() / (data.items**2 - data.items)

def mini_iita_fit(data: Dataset, qo: QuasiOrder):
    """
    Calculates the minimized IITA fit metric for a given dataset and quasiorder\n
    """
    p = np.nansum(data.rp.to_numpy(), axis=0, dtype=np.float64)

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

    p /= data.subjects

    expected_ce = np.zeros(data.ce.shape)

    for i in range(data.items):
        for j in range(data.items):
            if (i == j): continue

            if (qo.full_matrix[i][j]):
                expected_ce[i][j] = error * p[j] * data.subjects
            elif (not qo.full_matrix[j][i]):
                expected_ce[i][j] = (1 - p[i]) * p[j] * data.subjects
            else:
                expected_ce[i][j] = (p[j] * data.subjects) - ((p[i] - (p[i] * error)) * data.subjects)
    
    ce = data.ce.to_numpy().flatten()
    expected_ce = expected_ce.flatten()
    return ((ce - expected_ce) ** 2).sum() / (data.items**2 - data.items)