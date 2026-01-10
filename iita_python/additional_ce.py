import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import Self, List

from iita_python.dataset import Dataset

def pairwise_diff_ce(rp: pd.DataFrame) -> pd.DataFrame:
    """
    Computes counterexamples from a response pattern DataFrame by using pairwise differences of item correspondences\n
    Can be used for non-binary data because of not relying on strict 0/1 values\n
    Does not support missing values in the response patterns\n
    """

    items = rp.shape[1]
    subjects = rp.shape[0]

    ce = pd.DataFrame(0, index=np.arange(items), columns=np.arange(items))
    for i in range(subjects):
        # for subject i, if a < b, add b - a for all item pairs (a,b)
        # this is equivalent to ce[a][b] += 1 if a=0 and b=1, but works for non-binary data as well

        row = rp.loc[i].to_numpy()
        ce -= np.clip(row[:, None] - row[None, :], None, 0)

    return ce

def missing_value_substitution_ce(rp: pd.DataFrame) -> pd.DataFrame:
    """
    Computes counterexamples from a response pattern DataFrame by using pairwise differences of item correspondences\n
    Substitutes missing values in the response patterns with the mean of the item, making some counterexample amounts fractional\n
    Can be used for non-binary data because of not relying on strict 0/1 values\n
    """

    items = rp.shape[1]
    subjects = rp.shape[0]

    rp1 = rp.copy()

    for i in range(items):
        # substitute missing values in item i with the mean of the item
        col = rp1.loc[:, i].to_numpy()
        mean_val = np.nanmean(col)
        col = pd.Series(col).fillna(mean_val)
        rp1.loc[:, i] = col

    # then calculate pairwise difference counterexamples
    return pairwise_diff_ce(rp1)

def relativify(calculator: callable):
    """
    Decorator to relativify counterexample calculators\n
    The counterexample amounts are divided by the number of cases for each item pair where both items are not missing\n
    """

    def wrapper(rp: pd.DataFrame):
        f"""
        Computes counterexamples relative to the amount of valid cases using {calculator.__name__} as base calculator\n
        The counterexample amounts are divided by the number of cases for each item pair where both items are not missing\n
        """

        ce = calculator(rp)

        items = rp.shape[1]
        subjects = rp.shape[0]

        valid_cases = pd.DataFrame(0, index=np.arange(items), columns=np.arange(items))
        for i in range(subjects):
            #for subject i, increment all cases where neither a nor b are NaN (valid case for counterexamples)
            not_nan = np.logical_not(rp.loc[i].isna())
            valid_cases += np.outer(not_nan, not_nan).astype(int)

        # avoid division by zero
        valid_cases = valid_cases.replace(0, 1)

        return ce / valid_cases

    return wrapper

class AdditionalCEDataset(Dataset):
    def __init__(self, response_patterns: pd.DataFrame | npt.NDArray | List[List[int]]):
        """
        Computes the counterexamples and equivalence examples from response patterns\n
        Supports pandas dataframes, numpy arrays, and python lists\n
        Rows represent the subjects, columns - the items\n

        In addition to the base Dataset, this class provides additional counterexample calculators:\n
        - pairwise_diff_ce: computes counterexamples using pairwise differences of item correspondences, allowing for non-binary data\n
        - missing_value_substitution_ce: computes counterexamples using pairwise differences with missing values substituted by item means\n
        """
        super().__init__(response_patterns)

        self.pairwise_diff_ce = lambda self, relative=False: (
            relativify(pairwise_diff_ce) if relative else pairwise_diff_ce
        )(self.rp)
        self.pairwise_diff_ce.__doc__ = pairwise_diff_ce.__doc__

        self.missing_value_substitution_ce = lambda self, relative=False: (
            relativify(missing_value_substitution_ce) if relative else missing_value_substitution_ce
        )(self.rp)
        self.missing_value_substitution_ce.__doc__ = missing_value_substitution_ce.__doc__