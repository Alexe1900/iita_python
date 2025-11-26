import numpy as np
import numpy.typing as npt
from typing import Self, List
import pandas as pd

class Dataset():
    #aliases for response_patterns, counterexamples, equiv_examples
    @property
    def rp(self) -> pd.DataFrame:
        return self._rp
    @rp.setter
    def rp(self, inp: pd.DataFrame) -> None:
        self._rp = inp
    response_patterns = rp

    @property
    def ce(self) -> pd.DataFrame:
        return self._ce
    @ce.setter
    def ce(self, inp: pd.DataFrame) -> None:
        self._ce = inp
    counterexamples = ce

    @property
    def eqe(self) -> pd.DataFrame:
        return self._eqe
    @eqe.setter
    def eqe(self, inp: pd.DataFrame) -> None:
        self._eqe = inp
    equiv_examples = eqe

    @property
    def items(self):
        return self.ce.shape[0]
    
    @property
    def subjects(self):
        return self.rp.shape[0]
    
    @property
    def filled_vals(self):
        return (~np.isnan(self.rp)).sum(axis=0)

    def __init__(self, response_patterns: pd.DataFrame | npt.NDArray | List[List[int]]):
        """
        Computes the counterexamples and equivalence examples from response patterns\n
        Supports pandas dataframes, numpy arrays, and python lists\n
        Rows represent the subjects, columns - the items\n
        """
        self._rp = pd.DataFrame(response_patterns, index=None, columns=None)
        self._ce = None
        self._eqe = None
        
        #counterexamples computation   
        self.ce = pd.DataFrame(0, index=np.arange(self.rp.shape[1]), columns=np.arange(self.rp.shape[1]))

        for i in range(len(self.rp)):
            #for subject i, find all cases where a=0 and b=1 (counterexamples to b->a or a <= b) and increment where they intersect
            not_a = (self.rp.loc[i] == 0)
            b = (self.rp.loc[i] == 1)
            self.ce.loc[not_a, b] += 1
        
        #equivalence examples computation   
        self.eqe = pd.DataFrame(0, index=np.arange(self.rp.shape[1]), columns=np.arange(self.rp.shape[1]))
        for i in range(len(self.rp)):
            #for subject i, increment all cases where a=b (examples of equivalence of a and b)
            row = self.rp.loc[i].to_numpy()
            self.eqe += np.equal.outer(row, row).astype(int)
    
    def add(self, dataset_to_add: Self):
        """
        Add a second IITA_Dataset: concatenate the response patterns, add counterexamples and equivalence examples\n
        Item amounts must match, else ValueError
        """
        if (self.rp.shape[1] != dataset_to_add.shape[1]):
            raise ValueError('Item amounts must match')
        
        self.rp = pd.concat(self.rp, dataset_to_add.rp)
        self.ce = self.ce + dataset_to_add.ce
        self.eqe = self.eqe + dataset_to_add.eqe
    
    __iadd__ = add