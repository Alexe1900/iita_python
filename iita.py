import numpy as np
import numpy.typing as npt
import pandas as pd
from dataset import Dataset
from quasiorder import get_edge_list

def unfold_examples(
        matrix: pd.DataFrame,
        relativity: npt.NDArray | None = None,
        dtype=np.float32
    ) -> npt.NDArray:
    """
    Turns an item/item metric DataFrame into
    a list of tuples of the form (x, [i, j]), where matrix[i, j] = x.\n
    Can input a relativity matrix, then exery x gets divided by relativity[i, j].
    This can be used to account for missing values
    """

    dfmatrix = pd.DataFrame(matrix).astype(dtype)
    
    rel = relativity
    if (rel is None):
        rel = np.ones(dfmatrix.shape, dtype=int)
    
    dfmatrix = dfmatrix / rel

    n = dfmatrix.shape[0]
    pos = np.arange(n, dtype=np.int_)
    i = np.repeat(pos, n)
    j = np.tile(pos, n)
    res = np.array(list(zip(dfmatrix.to_numpy()[i, j], i, j)), dtype=np.int_)
    return res[res[:, 1] != res[:, 2]]

def orig_iita_fit(data: Dataset, qo):
    qo_edges = get_edge_list(qo)
    p = data.rp.to_numpy().sum(axis=0) / data.subjects

    error = 0
    for a, b in qo_edges:
        error += data.ce.iloc[a, b] / (p[b] * data.subjects)
    
    error /= len(qo_edges)

    expected_ce = np.zeros(data.ce.shape)

    for i in range(data.items):
        for j in range(data.items):
            if (i == j): continue

            if (qo[i][j]):
                expected_ce[i][j] = error * p[j] * data.subjects
            else:
                expected_ce[i][j] = (1 - p[i]) * p[j] * data.subjects * (1 - error)
    
    ce = data.ce.to_numpy().flatten()
    expected_ce = expected_ce.flatten()
    
    return ((ce - expected_ce) ** 2).sum() / (data.items**2 - data.items)