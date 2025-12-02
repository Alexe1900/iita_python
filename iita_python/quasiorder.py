import numpy as np
import numpy.typing as npt
import pandas as pd

class QuasiOrder:
    def __init__(self, matrix: npt.NDArray):
        self.full_matrix = matrix

    def get_edge_list(self, buff=0):
        """
        Returns the edge list of the quasiorder as a list of (a, b) pairs\n
        buff: int to add to each index (useful for 1-based indexing)
        """
        n = self.full_matrix.shape[0]
        edge_list = []

        for i in range(n):
            for j in range(n):
                if i == j: continue
                if (self.full_matrix[i][j]):
                    edge_list.append([i+buff, j+buff])
        
        return edge_list

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

def ind_gen(counterexamples: npt.NDArray, n: int) -> list[QuasiOrder]:
    """
    Inductively generates quasiorders from counterexample edge list\n
    Counterexamples is expected to be of the form returned by unfold_examples (array of (x, i, j) tuples)\n
    """
    ce = counterexamples

    if (len(ce) == 0): raise ValueError("Counterexamples can't be empty")

    ce = ce[ce[:, 0].argsort()]
    contracted_ce = [[]]
    for example in ce:
        if (len(contracted_ce[-1]) == 0): contracted_ce[-1].append(example)
        elif (contracted_ce[-1][0][0] == example[0]): contracted_ce[-1].append(example)
        else: contracted_ce.append([example])

    ce = [[ex[1:] for ex in g] for g in contracted_ce]

    qos = [np.eye(n, dtype=np.int_)]
    long_queue = np.empty((0, 2), dtype=np.int_)
    for group in ce:
        new_qo = qos[-1].copy()
        queue = np.concat([group, long_queue], axis=0)
        queue = np.array(sorted(queue.tolist()), dtype=np.int_)
        allow = np.ones((len(queue)))

        for a, b in queue:
            new_qo[a][b] = 1

        while (True):
            for i, (a, b) in enumerate(queue):
                for c in range(n):
                    if (c == a or c == b): continue

                    if (new_qo[b][c] and (not new_qo[a][c])) or (new_qo[c][a] and (not new_qo[c][b])):
                        new_qo[a][b] = 0
                        allow[i] = 0
                        break
            
            if (allow.sum() == len(allow)): break

            long_queue = queue[np.logical_not(allow)].copy()
            queue = queue[allow.astype(np.bool)].copy()
            allow = allow[allow.astype(np.bool)].copy()
        
        if (not (qos[-1] == new_qo).all()):
            qos.append(new_qo)
    
    return [QuasiOrder(qo) for qo in qos[1:]]