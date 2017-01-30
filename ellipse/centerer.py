from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from numpy import ma as ma


DEFAULT_THRESHOLD = 0.1
WINDOW_HALF_SIZE = 5

_in_mask = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1],
    [1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1],
    [1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1],
    [1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1],
    [1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1],
    [1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]
_out_mask = [
    [1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1],
    [1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1],
    [1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,1],
    [1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1],
    [1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,1],
    [1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1],
    [1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1],
]

class Centerer(object):
    '''
    Object centerer.

    '''
    def __init__(self, image, geometry, verbose=True):
        '''
        Object centerer.

        :param image: np 2-D array
            image array
        :param geometry: instance of Geometry
            geometry that directs the centerer to look at its X/Y
            coordinates. These are modified by the centerer algorithm.
        :param verbose: boolean, default True
            print object centering info
        '''
        self._image = image
        self._geometry = geometry
        self._verbose = verbose

        self.threshold = DEFAULT_THRESHOLD

        self._mask_half_size = len(_in_mask) / 2

        # number of pixels in each mask
        sz = len(_in_mask)
        self._ones_in  = ma.masked_array(np.ones(shape=(sz,sz)), mask=_in_mask)
        self._ones_out = ma.masked_array(np.ones(shape=(sz,sz)), mask=_out_mask)

        self._in_mask_npix = np.sum(self._ones_in)
        self._out_mask_npix = np.sum(self._ones_out)


    def center(self, threshold=DEFAULT_THRESHOLD):
        '''
        Runs the object centerer, eventually modifying in place
        the geometry associated with this Ellipse instance.

        :param threshold: float, default = 0.1
            object centerer threshold. To turn off the centerer, set this
            to a large value >> 1.
        '''
        if self._verbose:
            print("Centering on object....   ", end="")

        # Check if center coordinates point to somewhere inside the frame.
        # If not, set then to frame center.
        _x0 =  self._geometry.x0
        _y0 =  self._geometry.y0
        if _x0 is None or _x0 < 0 or _x0 >= self._image.shape[0] or \
           _y0 is None or _y0 < 0 or _y0 >= self._image.shape[1]:
            _x0 = self._image.shape[0] / 2
            _y0 = self._image.shape[1] / 2

        max_fom = 0.
        max_i = 0
        max_j = 0

        # scan all positions inside window
        for i in range(int(_x0 - WINDOW_HALF_SIZE), int(_x0 + WINDOW_HALF_SIZE) + 1):
            for j in range(int(_y0 - WINDOW_HALF_SIZE), int(_y0 + WINDOW_HALF_SIZE) + 1):

                # ensure that it stays inside image frame
                i1 = max(0, i - self._mask_half_size)
                j1 = max(0, j - self._mask_half_size)
                i2 = min(self._image.shape[0]-1, i + self._mask_half_size)
                j2 = min(self._image.shape[1]-1, j + self._mask_half_size)

                window = self._image[j1:j2,i1:i2]

                # averages in inner and outer regions.
                inner = ma.masked_array(window, mask=_in_mask)
                outer = ma.masked_array(window, mask=_out_mask)
                inner_avg = np.sum(inner) / self._in_mask_npix
                outer_avg = np.sum(outer) / self._out_mask_npix

                # standard deviation and figure of merit
                inner_std = np.std(inner)
                outer_std = np.std(outer)
                stddev = np.sqrt(inner_std**2 + outer_std**2)

                fom = (inner_avg - outer_avg) / stddev

                if fom > max_fom:
                    max_fom = fom
                    max_i = i
                    max_j = j

        # figure of merit > threshold: update geometry with new coordinates.
        if max_fom > threshold:
            self._geometry.x0 = float(max_i)
            self._geometry.y0 = float(max_j)

            if self._verbose:
                print("Done. Found x0 =%6.1f, y0 =%6.1f" % (self._geometry.x0, self._geometry.y0))
        else:
            if self._verbose:
                print("Done. Below threshold. Keeping original coordinates.")


