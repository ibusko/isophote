from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from scipy.optimize import leastsq


def harmonic_function(phi, y0, c):
    '''
    Compute harmonic function.

    :param phi: float or np.array
        angle(s) along the elliptical path, going counterclockwise,
        starting coincident with the position angle. That is, the
        angles are defined from the semi-major axis that lies in
        the +X quadrant.

    :param y0: float
        mean intensity
    :param y0: np array of shape (4)
        containing the four harmonic coefficients
    :return: float or np.array
        function value(s) at the given input angle(s)
    '''
    return y0 + c[0] * np.sin(phi) + c[1] * np.cos(phi) + c[2] * np.sin(2*phi) + c[3] * np.cos(2*phi)


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

    solution = leastsq(optimize_func, [np.mean(sample), a1, b1, a2, b2])

    if solution[1] > 4:
        raise RuntimeError("Error in least squares fit: " + solution[5])

    return solution[0]


