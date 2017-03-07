###############################################################################
# prop_mod.py
###############################################################################
#
# Function to properly calculate mod for floats
#
###############################################################################

import numpy as np

def pm_int(dividend, divisor):
    
    while (dividend >= divisor):
        dividend -= divisor

    while (dividend < 0.):
        dividend += divisor

    return dividend

pm = np.vectorize(pm_int)
