import numpy as np
from .utils import UnionFind
from .iita import unfold_examples

class QuasiOrder():
    n = None
    equiv = None
    comp_matrix = None

    @property
    def full_matrix(self):
        equiv_groups = self.equiv.get_groups()
        res = self.comp_matrix.copy()

        for group in equiv_groups:
            for i in group:
                for j in group:
                    res[i][j] = 1
        
        for i, group1 in enumerate(equiv_groups):
            for j, group2 in enumerate(equiv_groups):
                if (i == j): continue
                if (not self.comp_matrix[group1[0]][group2[0]]): continue

                for a in group1:
                    for b in group2:
                        res[a][b] = 1
        
        return res
    
    @property
    def edge_list(self):
        res = unfold_examples(self.full_matrix, dtype=np.int_)
        res = res[res[:, 0] == 1]
        res = [edge[1:] for edge in res]
        return res

    def __init__(self, n: int, adj_matrix = None, equiv_arr = None):
        self.n = n
        self.equiv = UnionFind(n, equiv_arr)

        self.comp_matrix = adj_matrix
        if (self.comp_matrix is None): self.comp_matrix = np.zeros((n, n))
    
    def apply_edge_list(self, edge_list):
        for a, b in edge_list:
            a = self.equiv.find(a)
            b = self.equiv.find(b)
            if (self.comp_matrix[a][b]): continue

            self.comp_matrix[a][b] = 1

            if (self.comp_matrix[b][a]): self.equiv.union(a, b)
    
    def copy(self):
        return QuasiOrder(self.n, self.comp_matrix.copy(), self.equiv.arr.copy())
    
    def __getitem__(self, idx):
        return self.comp_matrix[idx]

def generate_quasi_orders(counterexamples, n):
    """
    Inductively generates a list of quasi orders and equivalence classes from an array of counterexample relations\n
    Does not input a matrix, use unfold_examples to transform a matrix into a list\n
    Also intputs the amount of items
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

    qos = []

    active_qo = QuasiOrder(n)
    long_queue = np.empty((0, 2), dtype=np.int_)

    for group in ce:
        queue = np.concat([long_queue, group])
        allow = np.ones(len(queue))
        new_qo = active_qo.copy()

        while(True):
            new_qo = active_qo.copy()
            new_qo.apply_edge_list(queue)

            for i, (a, b) in enumerate(queue):
                a = new_qo.equiv.find(a)
                b = new_qo.equiv.find(b)
                for c in range(n):
                    c = new_qo.equiv.find(c)
                    if (c == a or c == b): continue

                    if (new_qo[b][c] and (not new_qo[a][c])): allow[i] = 0
                    if (new_qo[c][a] and (not new_qo[c][b])): allow[i] = 0
            
            if (allow.sum() == len(allow)): break

            long_queue = queue[np.logical_not(allow)].copy()
            queue = queue[allow.astype(np.bool)].copy()
            allow = allow[allow.astype(np.bool)].copy()
        
        if ((active_qo.full_matrix == new_qo.full_matrix).all()): continue

        active_qo = new_qo.copy()

        qos.append(active_qo.copy())
    
    return qos