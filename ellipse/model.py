from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from scipy.interpolate import UnivariateSpline

EPSILON = 1.e-8
SMOOTH = 1.e-8
MARGIN = 10


def build_model(image, isolist, background=0., high_harmonics=True, verbose=True):
    '''
    Builds model galaxy image from isophote list.

    The algorithm scans the input list, and, for each  ellipse in there, fills up the
    output image array with the corresponding isophotal intensity. Pixels in the target
    array are in general only partially covered by the isophote "pixel"; the algorithm
    takes care of this partial pixel coverage, by keeping track of how much intensity was
    added to each pixel by storing the partial area information in an auxiliary array.
    The information in this array is then used to normalize the pixel intensities.

    :param image: numpy 2-d array
        this array must be the same shape as the array used to generate the isophote list,
        so coordinates will match.
    :param isolist: IsophoteList instance
        the list created by class Ellipse
    :param background: float, default = 0.
        constant value to be added to each pixel
    :param high_harmonics: boolean, default True
        add higher harmonics (A3,B3,A4,B4) to result?
    :param verbose: boolean, default True
        print info
    :return: numpy 2-D array
        with the model image
    '''
    result = np.zeros(shape=image.shape)

    # a small amount of smoothing seems necessary for the interpolator
    # to behave properly. We may have to test this under a wider
    # variety of circumstances.
    smoothing_factor = SMOOTH *len(isolist.sma)

    # the target grid is spaced in 0.1 pixel intervals so as
    # to ensure no gaps will result on the output array.
    finely_spaced_sma = np.arange(isolist[0].sma, isolist[-1].sma, 0.1)

    if verbose:
        print("Interpolating....", end="")

    # interpolate ellipse parameters
    interpolator = UnivariateSpline(isolist.sma, isolist.intens, s=smoothing_factor)
    intens = interpolator(finely_spaced_sma)

    interpolator = UnivariateSpline(isolist.sma, isolist.eps, s=smoothing_factor)
    eps = interpolator(finely_spaced_sma)

    interpolator = UnivariateSpline(isolist.sma, isolist.pa, s=smoothing_factor)
    pa = interpolator(finely_spaced_sma)

    interpolator = UnivariateSpline(isolist.sma, isolist.x0, s=smoothing_factor)
    x0 = interpolator(finely_spaced_sma)

    interpolator = UnivariateSpline(isolist.sma, isolist.y0, s=smoothing_factor)
    y0 = interpolator(finely_spaced_sma)

    # discard central point since these attributes are None at the center.
    interpolator = UnivariateSpline(isolist.sma[1:], isolist.grad[1:], s=smoothing_factor)
    grad = interpolator(finely_spaced_sma)

    interpolator = UnivariateSpline(isolist.sma[1:], isolist.a3[1:], s=smoothing_factor)
    a3 = interpolator(finely_spaced_sma)

    interpolator = UnivariateSpline(isolist.sma[1:], isolist.b3[1:], s=smoothing_factor)
    b3 = interpolator(finely_spaced_sma)

    interpolator = UnivariateSpline(isolist.sma[1:], isolist.a4[1:], s=smoothing_factor)
    a4 = interpolator(finely_spaced_sma)

    interpolator = UnivariateSpline(isolist.sma[1:], isolist.b4[1:], s=smoothing_factor)
    b4 = interpolator(finely_spaced_sma)

    if verbose:
        print("Done")

    # for each interpolated isophote, generate intensity values
    # on the output image array
    for index in range(len(finely_spaced_sma)):
        pass











    # add background *after* model image was added *and* normalized,
    # otherwise normalization will be wrong.
    result += background

    return result

