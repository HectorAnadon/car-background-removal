import numpy as np

def rlencode(x, dropna=False):
    """
    Run length encoding.
    Based on http://stackoverflow.com/a/32681075, which is based on the rle 
    function from R.
    
    Parameters
    ----------
    x : 1D array_like
        Input array to encode
    dropna: bool, optional
        Drop all runs of NaNs.
    
    Returns
    -------
    start positions, run lengths, run values
    
    """
    where = np.flatnonzero
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return (np.array([], dtype=int), 
                np.array([], dtype=int), 
                np.array([], dtype=x.dtype))

    starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]
    
    if dropna:
        mask = ~np.isnan(values)
        starts, lengths, values = starts[mask], lengths[mask], values[mask]
    
    return starts, lengths, values