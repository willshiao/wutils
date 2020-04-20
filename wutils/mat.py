"""
Matrix-related functions and classes.
Currently only contains the MarkedMatrix class.
"""

from collections import OrderedDict
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

class MarkedMatrix:
    """
    A wrapper around a 2D numpy array that includes labels for each
    segment of the matrix.

    Also includes several functions to display the data.
    """

    def __init__(self, items):
        """Initializes a MarkedMatrix.
        Can take one of two types of arguments:
            1. A `list` or `tuple` of label-array tuples.
            2. A `tuple` consisting of a
                (0) data matrix (np.array)
                    - can be obtained from a MarkedMatrix with `get_mat()`
                (1) location index (OrderedDict)

        Arguments:
            items {list|tuple} -- As described above.
        """
        # Construct a MarkedMatrix from a list/tuple of lists/tuples of labels to data.
        # TODO: support all iterable types (can't use indexing)
        if isinstance(items[0], tuple) or isinstance(items[0], list):
            self.mat = np.vstack([item[1] for item in items])
            self.loc_idx = OrderedDict()
            total = 0
            for item in items:
                rows = item[1].shape[0]
                total += rows
                self.loc_idx[total] = item[0]

        # Construct a MarkedMatrix from an existing data matrix and label index
        else:
            self.mat = items[0]
            self.loc_idx = items[1]

    def get_mat(self):
        """Returns the internal matrix.
        Identical to the `mat` property.

        Returns:
            np.array -- The internal matrix of all rows in the MarkedMatrix.
        """
        return self.mat

    def get_loc_idx(self):
        """Returns the internal location matrix mapping
        the last row number to a label.

        Returns:
            OrderedDict[int, str] -- A location index.
                If a row number is less than the key, that row belongs to the
                label equal to the current value.
        """
        return self.loc_idx

    def tsne(self, tsne_args=None, ax=None, plot=True):
        """Runs TSNE on the matrix and plots the results using seaborn
        with the labels defined at the construction of the MarkedMatrix.

        Keyword Arguments:
            tsne_args {dict} -- An optional dictionary of the arguments to be passed
                to sklearn's TSNE constructor (default {None})
            ax {matplotlib.pyplot.Axis} -- An optional Axis object used to draw the plot.
                (default: {None})
            plot {bool} -- Whether or not the function should draw the plot. (default: {True})

        Returns:
            np.array -- An M x 2 numpy array of the original matrix projected
                into 2 dimensions using TSNE.
        """
        # Forward TSNE args
        if tsne_args is not None:
            tsne = TSNE(n_components=2, **tsne_args)
        else:
            tsne = TSNE(n_components=2)
        M = tsne.fit_transform(self.mat)
        # Keeps track of the start of the current section
        last_loc = 0
        if plot:
            for loc, name in self.loc_idx.items():
                sns.scatterplot(x=M[last_loc:loc, 0], y=M[last_loc:loc, 1], label=name, ax=ax)
                last_loc = loc
        return M

    def get_pieces(self, mat=None):
        """Given a matrix, returns a dictionary of labels to their corresponding submatrices.
        Returns a mapping for the MarkedMatrix if no matrix is provided.

        Keyword Arguments:
            mat {np.array} -- The input matrix (default: {None})

        Returns:
            dict[str, np.array] -- A mapping from labels to matrices.
        """
        if mat is None:
            mat = self.mat
        out = OrderedDict()
        last_loc = 0
        for loc, name in self.loc_idx.items():
            out[name] = mat[last_loc:loc]
            last_loc = loc
        return out
