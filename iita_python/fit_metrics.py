import numpy as np
import numpy.typing as npt
import pandas as pd
from .dataset import Dataset
from .quasiorder import QuasiOrder

def orig_iita_fit(data: Dataset, qo: QuasiOrder):
    qo_edges = qo.get_edge_list()
    p = data.rp.to_numpy().sum(axis=0) / data.subjects

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