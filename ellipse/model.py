from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np


EPSILON = 1.e-8
MARGIN = 10

def build_model(image, isophote_list, background=0.):
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
    :param isophote_list: IsophoteList instance
        the list created by class Ellipse
    :param background: float, default = 0.
        constant value to be added to each pixel
    :return: numpy 2-D array
        with the model image
    '''
    result = np.zeros(shape=image.shape)








    # add background *after* model image was added *and* normalized,
    # otherwise normalization will be wrong.
    result += background

    return result
