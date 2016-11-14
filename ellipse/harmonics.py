from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from scipy.optimize import leastsq


def harmonic_function(phi, y0, a1, b1, a2, b2):
    '''
    Compute harmonic function.

    :param phi: float or np.array
        angle(s) along the elliptical path, going counterclockwise,
        starting coincident with the position angle. That is, the
        angles are defined from the semi-major axis that lies in
        the +X quadrant.

    :param y0: float
        mean intensity
    :param a1: float
        first harmonic coefficient
    :param b1: float
        2nd harmonic coefficient
    :param a2: float
        3rd harmonic coefficient
    :param b2: float
        4th harmonic coefficient
    :return: float or np.array
        function value(s) at the given input angle(s)
    '''
    return y0 + a1 * np.sin(phi) + b1 * np.cos(phi) + a2 * np.sin(2*phi) + b2 * np.cos(2*phi)


def fit_harmonics(phi, sample):
    '''
    Fit harmonic function to a set of angle,intensity pairs

    :param phi: np.array
        angles defined in the same way as in harmonic_function
    :param sample: np.array
        intensities samples along the elliptical path, at the angles defined in parameter 'phi'
    :return: 5 float values
        fitted values for y0, a1, b1, a2, b2
    '''
    a1 = 1.
    b1 = 1.
    a2 = 1.
    b2 = 1.
    optimize_func = lambda x: x[0] + x[1]*np.sin(phi) + x[2]*np.cos(phi) + x[3]*np.sin(2*phi) + x[4]*np.cos(2*phi) - sample
    return leastsq(optimize_func, [np.mean(sample), a1, b1, a2, b2])[0]


