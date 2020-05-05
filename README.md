# wutils
[![](https://img.shields.io/pypi/v/wutils.svg)](https://pypi.org/pypi/wutils/)

A collection of Python utility functions/classes.

Currently only provides one class: `wutils.mat.MarkedMatrix`, which is a wrapper around a `numpy.array` with additional row labels and helper functions.

Can be installed with `pip install wutils`.

## Example

```python
import numpy as np
import matplotlib.pyplot as plt
from wutils.mat import MarkedMatrix

# Create a MarkedMatrix from a tuple of label-matrix tuples
mm = MarkedMatrix((
    ('a', np.random.randn(100, 100)),
    ('b', np.random.randn(50, 100)) # num. of columns must match 'a'
))

# Create a labeled TSNE plot of the components
plt.figure()
mm.tsne()
plt.show()

# Perform SVD on the full matrix
U, _, _ = np.linalg.svd(mm.get_mat(), full_matrices=False)

# Form another MarkedMatrix consisting of the first 2 columns of U.
# We reuse our existing labels
mm_U = MarkedMatrix((U[:, 2], mm.get_loc_idx()))

# Split up the MarkedMatrix to an OrderedDict of label to submatrix
print(mm_U.get_pieces())

# Output:
# OrderedDict([
#     ('a', array([...])),
#     ('b', array([...]))
# ])

# where 'a' is 100 x 2
# and 'b' is 50 x 2
```
