import numpy as np
import pandas as pd

def general_ce(rp: pd.DataFrame):
    """
    Computes counterexamples from a response pattern DataFrame\n
    Relies on strict 0/1 values in the response patterns\n
    Supports missing values in the response patterns by ignoring them\n
    """
    items = rp.shape[1]
    subjects = rp.shape[0]

    ce = pd.DataFrame(0, index=np.arange(items), columns=np.arange(items))
    for i in range(subjects):
        #for subject i, increment all cases where a=0 and b=1 (counterexamples to b->a or a <= b)
        not_a = (rp.loc[i] == 0)
        b = (rp.loc[i] == 1)
        ce.loc[not_a, b] += 1
    
    return ce

def subtraction_ce(rp: pd.DataFrame):
    """
    Computes counterexamples from a response pattern DataFrame by using differences of item pairs\n
    Can be used for non-binary data because of not relying on strict 0/1 values\n
    Does not support missing values in the response patterns\n
    """
    items = rp.shape[1]
    subjects = rp.shape[0]

    ce = pd.DataFrame(0, index=np.arange(items), columns=np.arange(items))
    for i in range(subjects):
        # for subject i, if a < b, add b - a for all pairs (a,b)
        # this is equivalent to ce[a][b] += 1 if a=0 and b=1, but works for non-binary data as well

        row = rp.loc[i].to_numpy()
        ce -= np.clip(row[:, None] - row[None, :], None, 0)

    return ce

def missing_value_substitution_ce(rp: pd.DataFrame):
    """
    Computes counterexamples from a response pattern DataFrame by using differences of item pairs\n
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

    return subtraction_ce(rp1)