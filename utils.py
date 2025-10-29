import numpy as np
import pandas as pd
import os

def read_rp(
        filename: str,
        nan_vals: list = [],
        separator: str = ',',
        excel_sheet_id: int = 0
    ) -> pd.DataFrame:
    """
    Reads a list of response patterns from a file\n
    Supports all pandas-readable datatypes and .npy\n
    Rows represent the respondents, columns - the items\n
    Values in nan_vals get replaced by NaN in the data\n
    """

    #filename checks
    if (not os.path.isfile(filename)):
        raise ValueError('Invalid filename')
    if (not os.access(filename, os.R_OK)):
        raise ValueError('Unreadable file')
    
    #response pattern reading
    rp = None
    if (filename[-3:] == 'xls' or filename[-4:] == 'xlsx'): #excel
        rp = pd.read_excel(filename, sheet_name=excel_sheet_id, header=None, na_values=nan_vals)
    elif (filename[-3:] == 'npy'): #npy
        rp = pd.DataFrame(np.load(filename))

        rp[rp in nan_vals] = np.nan
    else: #sonstiges
        rp = pd.read_table(filename, sep=separator, header=None, na_values=nan_vals)
    
    return rp