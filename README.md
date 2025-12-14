# IITA_Python

A Python implementation of the Inductive ITem Tree Analysis (IITA) algorithm for analyzing and validating quasi-orderings in psychometric data.

Intended to replicate the functionality DAKS package from R, with an OOP-style interface for simpler functionality expansion

## Installation

### From PyPI

```bash
pip install iita_python
```

## Quick Start

```python
from iita_python import Dataset, ind_gen, unfold_examples, orig_iita_fit
import iita_python.utils as utils

# Load response patterns from CSV
response_patterns = utils.read_rp('data.csv')

# Create Dataset: computes counterexamples and equivalence examples
data = Dataset(response_patterns)

# Extract counterexamples and generate quasi-orderings
ce = unfold_examples(data.ce)
qos = ind_gen(ce, data.items)

# Evaluate fit for each quasi-order
for i, qo in enumerate(qos):
    fit = orig_iita_fit(data, qo)
    print(f"Quasi-order {i}: fit = {fit:.2f}")
```

## Data Format

### Input: Response Patterns

Response patterns should be a 2D array where:
- **Rows** represent subjects (respondents)
- **Columns** represent items (questions/tasks)
- **Values** are 0 (incorrect) or 1 (correct), with NaN for missing responses

Example (CSV):
```
1,0,1,0,1
0,0,1,0,1
1,1,1,1,1
```

When reading from a file with `utils.read_rp()`, missing data can be specified via the `nan_vals` parameter.

## Core Modules

### `dataset.py`

**`Dataset` class**

Stores response patterns and computes derived metrics:

- `rp`: response patterns (DataFrame)
- `ce`: counterexamples - pairs (i, j) where subject has item i incorrect but item j correct
- `eqe`: equivalence examples - pairs (i, j) where subject answered items i and j identically
- `items`: number of items
- `subjects`: number of subjects
- `filled_vals`: number of non-missing responses per item

### `quasiorder.py`

**`unfold_examples(matrix, relativity=None, dtype=np.float32)`**

Converts a 2D matrix (e.g., counterexamples or equivalence examples) into a list of (value, i, j) tuples, excluding diagonal entries. Optionally normalizes by a relativity matrix.

**`ind_gen(counterexamples, n)`**

Generates candidate quasi-orderings from counterexample data. Returns a list of quasi-order matrices (numpy arrays) that progressively include edges.

**`get_edge_list(qo_matrix, buff=0)`**

Extracts the edge list from a quasi-order matrix as a list of (i, j) pairs.

### `fit_metrics.py`

**`orig_iita_fit(data, qo)`**

Computes the fit of a quasi-order to observed data using Schrepp's method:

1. Estimates an error rate from counterexamples on edges in the quasi-order
2. Predicts expected counterexamples for all item pairs under the quasi-order
3. Computes mean squared error between observed and expected counterexamples

Returns: float (MSE, lower is better)

## Requirements

- Python >= 3.8
- numpy
- pandas

## Testing

See the `test_ipynbs` folder. You can open the Jupyter notebooks in Google Colab and run all cells to see test results.

I am comparing my results on the PISA dataset to those of Milan Segedinac ([his implementation](https://github.com/milansegedinac/kst))

Please report any test failures in an issue

## Contributing

Pull requests and issues are welcome. For major changes, please open an issue first to discuss.

## IITA Overview

Schrepp (1999, 2003) developed IITA (Inductive itemm Tree Analysis) as a means to derive a surmise relation from dichotomous response patterns. Sargin and Ünlü (2009; Ünlü & Sargin, 2010) implemented two advanced versions of that procedure.

The three inductive item tree analysis algorithms are exploratory methods for extracting quasi orders (surmise relations) from data. In each algorithm, competing binary relations are generated (in the same way for all three versions), and a fit measure (differing from version to version) is computed for every relation of the selection set in order to find the quasi order that fits the data best. In all three algorithms, the idea is to estimate the numbers of counterexamples for each quasi order, and to find, over all competing quasi orders, the minimum value for the discrepancy between the observed and expected numbers of counterexamples.

The three data analysis methods differ in their choices of estimates for the expected numbers of counterexamples. (For an item pair (i,j), the number of subjects solving item j but failing to solve item i, is the corresponding number of counterexamples. Their response patterns contradict the interpretation of (i,j) as `mastering item j implies mastering item i.')

## References

- Schrepp, M. (2001). IITA: A program for the analysis of individual item and step matrices. Unpublished technical report.
- Knowledge Space Theory: https://en.wikipedia.org/wiki/Knowledge_space

## Author

Alexe1900, mentored and supervised by Peter Steiner from PHSG St. Gallen

---

## Roadmap

- [ ] Full DAKS functionality
- [ ] Performance optimizations for large datasets
- [ ] Visualization tools for quasi-orderings
- [ ] Comprehensive test suite (unit + integration)
