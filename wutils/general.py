"""
General-purpose miscellaneous functions.
"""
import pickle

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
