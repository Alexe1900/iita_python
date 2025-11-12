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

class UnionFind():
    arr = []
    n = 0

    def __init__(self, n: int, arr = None):
        self.n = n

        self.arr = arr
        if (self.arr is None): self.arr = list(range(n))
    
    def find(self, x):
        if (self.arr[x] != x): self.arr[x] = self.find(self.arr[x])
        return self.arr[x]
    
    def union(self, x, y):
        xr = self.find(x)
        yr = self.find(y)

        self.arr[xr] = yr
    
    def get_unfiltered_groups(self):
        groups = []

        for i in range(self.n): groups.append([])
        for i in range(self.n):
            groups[self.find(i)].append(i)
        
        return groups
    
    def get_groups(self):
        
        return list(filter(len, self.get_unfiltered_groups()))
    
    def copy(self):
        return UnionFind(self.n, self.arr.copy())