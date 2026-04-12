import numpy as np
import numpy.typing as npt
from typing import Self, List
import pandas as pd

class Dataset():
    # full name aliases for response patterns, counterexamples and equivalence examples
    @property
    def response_patterns(self) -> pd.DataFrame:
        return self.rp
    @response_patterns.setter
    def response_patterns(self, inp: pd.DataFrame) -> None:
        self.rp = inp

    @property
    def counterexamples(self) -> pd.DataFrame:
        return self.ce
    @counterexamples.setter
    def counterexamples(self, inp: pd.DataFrame) -> None:
        self.ce = inp

    @property
    def equiv_examples(self) -> pd.DataFrame:
        return self.eqe
    @equiv_examples.setter
    def equiv_examples(self, inp: pd.DataFrame) -> None:
        self.eqe = inp

    @property
    def items(self):
        return self.rp.shape[1]
    
    @property
    def subjects(self):
        return self.rp.shape[0]
    
    @property
    def filled_vals(self):
        return (np.logical_not(np.isnan(self.rp)).astype(int)).sum(axis=0)

    def __init__(self, response_patterns: pd.DataFrame | npt.NDArray | List[List[int]]):
        """
        Computes counterexamples, equivalence examples and valid CE cases from response patterns\n
        Supports pandas dataframes, numpy arrays, and python lists\n
        Rows represent the subjects, columns - the items\n
        """
        self.rp = pd.DataFrame(response_patterns)
        self.ce = None
        self.eqe = None
        
        rp_numpy = self.rp.to_numpy()

        # setting missing values (NaN) to 0
        rp_no_nan = np.nan_to_num(rp_numpy, 0) # NaN to 0
        not_rp_no_nan = np.logical_not(np.nan_to_num(rp_numpy, 1)).astype(int) # NaN to 1, negated to 0

        # counterexamples computation
        # all cases where a=0 and b=1 (counterexamples to b->a or a <= b)
        self.ce = pd.DataFrame(not_rp_no_nan.T @ rp_no_nan, index=self.rp.columns, columns=self.rp.columns)
        
        # equivalence examples computation
        # all cases where a=b, equivalent to ((a and b) or (~a and ~b))
        a_and_b = rp_no_nan.T @ rp_no_nan
        not_a_and_not_b = (not_rp_no_nan).T @ (not_rp_no_nan)
        self.eqe = pd.DataFrame(a_and_b + not_a_and_not_b, index=self.rp.columns, columns=self.rp.columns)

        # valid CE cases computation
        rp_isnt_nan = np.logical_not(np.isnan(rp_numpy)).astype(int)
        self.valid_ce_cases = pd.DataFrame(rp_isnt_nan.T @ rp_isnt_nan, index=self.rp.columns, columns=self.rp.columns)
    
    def add(self, dataset_to_add: Self):
        """
        Add a second IITA_Dataset: concatenate the response patterns, add CE, EQE and valid CE cases\n
        Item amounts must match, else ValueError
        """
        if (self.items != dataset_to_add.items):
            raise ValueError('Item amounts must match')
        
        self.rp = pd.concat(self.rp, dataset_to_add.rp)
        self.ce = self.ce + dataset_to_add.ce
        self.eqe = self.eqe + dataset_to_add.eqe
        self.valid_ce_cases = self.valid_ce_cases + dataset_to_add.valid_ce_cases

    @property 
    def relative_ce(self) -> pd.DataFrame:
        """
        Returns the counterexamples matrix accounting for missing values
        """
        return self.ce / self.valid_ce_cases
    
    __iadd__ = add