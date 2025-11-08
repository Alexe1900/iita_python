import numpy as np
import numpy.typing as npt
import pandas as pd

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