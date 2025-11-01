import numpy as np
import numpy.typing as npt
from typing import Self, List, Tuple
import pandas as pd

def unfold_examples(
        matrix: pd.DataFrame | npt.NDArray[np.int_],
        relativity: npt.NDArray[np.int_] | None = None
    ) -> npt.NDArray[np.float32]:
    """
    Turns an item/item metric DataFrame or matrix into
    a list of tuples of the form (x, [i, j]), where matrix[i, j] = x.\n
    Can input a relativity matrix, then exery x gets divided by relativity[i, j].
    This can be used to account for missing values
    """

    dfmatrix = pd.DataFrame(matrix).astype(np.float32)
    
    rel = relativity
    if (rel is None):
        rel = np.ones(dfmatrix.shape, dtype=int)
    
    dfmatrix = dfmatrix / rel

    itemnames = dfmatrix.columns.values

    n = len(itemnames)
    pos = np.arange(n)
    i = np.repeat(itemnames, n)
    j = np.tile(itemnames, n)
    res = np.array(list(zip(dfmatrix.to_numpy()[np.repeat(pos, n), np.tile(pos, n)], i, j)))
    return res[res[:, 1] != res[:, 2]]