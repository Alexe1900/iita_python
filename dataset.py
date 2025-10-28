import numpy as np
import numpy.typing as npt
from typing import Self
import pandas as pd
import os

class IITA_Dataset():
    #to use both response_patterns and rp
    @property
    def rp(self) -> pd.DataFrame:
        return self._rp
    @rp.setter
    def rp(self, inp: pd.DataFrame) -> None:
        self._rp = inp
    response_patterns = rp

    #to use both counterexamples and ce
    @property
    def ce(self) -> pd.DataFrame:
        return self._ce
    @ce.setter
    def ce(self, inp: pd.DataFrame) -> None:
        self._ce = inp
    counterexamples = ce

    #to use both equiv_examples and eqe
    @property
    def eqe(self) -> pd.DataFrame:
        return self._eqe
    @eqe.setter
    def eqe(self, inp: pd.DataFrame) -> None:
        self._eqe = inp
    equiv_examples = eqe

    def __init__(self, response_patterns: pd.DataFrame | npt.NDArray):
        """
        Computes the counterexamples and equivalence examples from response patterns\n
        Supports pandas dataframes, numpy arrays, and python lists\n
        Rows represent the respondents, columns - the items\n
        """
        self._rp = pd.DataFrame(response_patterns, index=None, columns=None)
        self._ce = None
        self._eqe = None
        
        #counterexamples computation   
        self.ce = pd.DataFrame(0, index=np.arange(self.rp.shape[1]), columns=np.arange(self.rp.shape[1]))

        for i in range(len(self.rp)):
            #for respondent i, find all cases where a=0 and b=1 (counterexamples to b->a or a <= b) and increment where they intersect
            not_a = (self.rp.loc[i] == 1)
            b = (self.rp.loc[i] == 0)
            self.ce.loc[not_a, b] += 1
        
        #equivalence examples computation   
        self.eqe = pd.DataFrame(0, index=np.arange(self.rp.shape[1]), columns=np.arange(self.rp.shape[1]))
        for i in range(len(self.rp)):
            #for respondent i, increment all cases where a=b (examples of equivalence of a and b)
            row = self.rp.loc[i].to_numpy()
            self.eqe += np.equal.outer(row, row).astype(int)
    
    def add(self, dataset_to_add: Self) -> Self:
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