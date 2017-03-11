###############################################################################
# prop_mod.py
###############################################################################
# HISTORY:
#   2017-03-11 - Written - Nick Rodd (MIT)
###############################################################################

import numpy as np

def pm_int(dividend, divisor):
    """
    Function to properly calculate mod for floats
    
    Returns:
        dividend mod divisor
    """
    
    while (dividend >= divisor):
        dividend -= divisor

    while (dividend < 0.):
        dividend += divisor

    return dividend

pm = np.vectorize(pm_int)
