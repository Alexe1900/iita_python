import numpy as np
import pandas as pd
import os

class IITA_Dataset():
    #to use both response_patterns and rp
    @property
    def rp(self):
        return self._rp
    @rp.setter
    def rp(self, inp):
        self._rp = inp
    response_patterns = rp

    #to use both counterexamples and ce
    @property
    def ce(self):
        return self._ce
    @ce.setter
    def ce(self, inp):
        self._ce = inp
    counterexamples = ce

    #to use both equiv_examples and eqe
    @property
    def eqe(self):
        return self._eqe
    @ce.setter
    def eqe(self, inp):
        self._eqe = inp
    equiv_examples = eqe

    def __init__(self, filename, nan_vals=[], separator=',', excel_sheet_id=0):
        """
        Initializes a list of response patterns from a file,
        along with computing the counterexamples and equivalence examples\n
        Supports all pandas-readable datatypes and .npy\n
        Rows must represent the respondents, columns - the items\n
        Values in nan_vals get replaced by NaN in the data\n
        """
        self._rp = None
        self._ce = None
        self._eqe = None

        #filename checks
        if (not os.path.isfile(filename)):
            raise ValueError('Invalid filename')
        if (not os.access(filename, os.R_OK)):
            raise ValueError('Unreadable file')
        
        #response pattern reading
        if (filename[-3:] == 'xls' or filename[-4:] == 'xlsx'):
            self.rp = pd.read_excel(filename, sheet_name=excel_sheet_id, header=None, na_values=nan_vals)
        elif (filename[-3:] == 'npy'):
            self.rp = pd.DataFrame(np.load(filename))

            self.rp[self.rp in nan_vals] = np.nan
        else:
            self.rp = pd.read_table(filename, sep=separator, header=None, na_values=nan_vals)
        
        #counterexamples computation   
        self.ce = pd.DataFrame(0, index=np.arange(len(self.rp)), columns=np.arange(len(self.rp)))

        for i in range(len(self.rp)):
            #for respondent i, find all cases where a=1 and b=0 (counterexamples to a->b) and increment where they intersect
            a = self.rp.loc[i] == 1
            not_b = self.rp.loc[i] == 0
            self.ce.loc[a, not_b] += 1
        
        #equivalence examples computation   
        for i in range(len(self.rp)):
            #for respondent i, increment all cases where a=b (examples of equivalence of a and b)
            row = self.rp.loc[i]
            self.eqe.loc[np.equal.outer(row, row)] += 1