import numpy as np
import numpy.typing as npt
import pandas as pd
from .dataset import Dataset
from .quasiorder import QuasiOrder
from .fit_segments import *

def iita_decorator(error_rate_calc, expected_ce_calc, compare = avg_squared_diff):
    """
    A decorator to create IITA fit metric functions with different error rate, expected counterexamples, and comparison calculations\n
    error_rate has to take Dataset and QuasiOrder as arguments and return a tuple (p, error)\n
    expected_ce has to take Dataset, QuasiOrder, p and error as arguments and return the expected counterexamples matrix\n
    compare has to take two matrices and return a fit metric value\n
    """
    def calc(data: Dataset, qo: QuasiOrder):
        p, error = error_rate_calc(data, qo)

        expected_ce = expected_ce_calc(data, qo, p, error)
        
        return compare(data.ce, expected_ce)
    
    return calc

orig_iita_fit = iita_decorator(orig_error_rate, orig_expected_ce)
orig_iita_fit.__doc__ = "Calculates the original IITA fit metric for a given dataset and quasiorder"

corr_iita_fit = iita_decorator(orig_error_rate, corr_expected_ce)
corr_iita_fit.__doc__ = "Calculates the corrected IITA fit metric for a given dataset and quasiorder"

mini_iita_fit = iita_decorator(mini_error_rate, corr_expected_ce)
mini_iita_fit.__doc__ = "Calculates the minimized IITA fit metric for a given dataset and quasiorder"