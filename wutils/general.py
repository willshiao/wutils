"""
General-purpose miscellaneous functions.
"""
import pickle
import numpy as np

def save_pickle(obj, loc, protocol=pickle.HIGHEST_PROTOCOL):
    """Saves a pickled version of `obj` to `loc`.
    Allows for quick 1-liner to save a pickle without leaving a hanging file handle.
    Useful for Jupyter notebooks.

    Also behaves differently to pickle in that it defaults to pickle.HIGHEST_PROTOCOL
        instead of pickle.DEFAULT_PROTOCOL.

    Arguments:
        obj {Any} -- The object to be pickled.
        loc {Path|str} -- A location to save the object to.
        protocol {pickle.Protocol} -- The pickle protocol level to use.
            (default {pickle.HIGHEST_PROTOCOL})
    """
    with open(loc, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)

def load_pickle(loc):
    """Loads a pickled object from `loc`.
    Very helpful in avoiding overwritting a pickle when you read using 'wb'
    instead of 'rb' at 3 AM.
    Also provides a convinient 1-liner to read a pickle without leaving an open file handle.

    Arguments:
        loc {Path|str} -- A location to read the pickled object from.

    Returns:
        Any -- The pickled object.
    """
    with open(loc, 'rb') as f:
        return pickle.load(f)

def sample_mat(mat, n, replace=False, return_idxs=False):
    """Sample `n` random rows from mat.
    Returns an array of the rows sampled from the matrix.

    Arguments:
        mat {np.array} -- A matrix to sample rows from.
        n {int} -- The number of rows to sample.

    Keyword Arguments:
        replace {bool} -- Whether or not to sample with replacement. (default: {False})
        return_idxs {bool} -- Whether or not to return the indexes of the selected rows. (default: {False})

    Returns:
        np.array|tuple -- Returns a matrix containing the sampled rows, or if `return_idxs` is set,
            returns a tuple containing the indexes of the selected rows and the sampled matrix.
    """    
    row_idxs = np.random.choice(mat.shape[0], n, replace=replace)
    sampled = mat[row_idxs, :]
    if return_idxs:
        return (row_idxs, sampled)
    return sampled
