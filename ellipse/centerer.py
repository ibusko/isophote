from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from numpy import ma as ma


DEFAULT_THRESHOLD = 1.0


class Centerer(object):
    '''
    Object centerer.

    '''
    def __init__(self, image, geometry):
        '''
        Object centerer.

        :param image: np 2-D array
            image array
        :param geometry: instance of Geometry
            geometry that directs the centerer to look at its X/Y
            coordinates. These are modified by the centerer algorithm.
        '''
        self._image = image
        self._geometry = geometry

        self.threshold = DEFAULT_THRESHOLD

        self._in_mask = [
            [0,0,0,0,0, 0,0,0,0,0],
            [0,0,0,0,0, 0,0,0,0,0],
            [0,0,0,0,1, 1,0,0,0,0],
            [0,0,0,1,0, 0,1,0,0,0],
            [0,0,1,0,0, 0,0,1,0,0],

            [0,0,1,0,0, 0,0,1,0,0],
            [0,0,0,1,0, 0,1,0,0,0],
            [0,0,0,0,1, 1,0,0,0,0],
            [0,0,0,0,0, 0,0,0,0,0],
            [0,0,0,0,0, 0,0,0,0,0],
        ]
        # self._in_mask = [
        #     [0,0,0,0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0,0],
        #     [0,0,0,1,1,1,1,0,0,0],
        #     [0,0,0,1,1,1,1,0,0,0],
        #     [0,0,0,1,1,1,1,0,0,0],
        #     [0,0,0,1,1,1,1,0,0,0],
        #     [0,0,0,0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0,0],
        # ]
        self._out_mask = [
            [0,0,0,1,1,1,1,0,0,0],
            [0,0,1,0,0,0,0,1,0,0],
            [0,1,0,0,0,0,0,0,1,0],
            [1,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,1],
            [0,1,0,0,0,0,0,0,1,0],
            [0,0,1,0,0,0,0,1,0,0],
            [0,0,0,1,1,1,1,0,0,0],
        ]

        self._in_mask_npix = np.sum(np.array(self._in_mask))
        self._out_mask_npix = np.sum(np.array(self._out_mask))

    def center(self, threshold=DEFAULT_THRESHOLD):
        '''
        Runs the object centerer, modifying in place the geometry
        associated with this Ellipse instance.

        :param threshold: float, default = 1.0
            object centerer threshold. To turn off the centerer, set this to zero.
        '''
        # Check if center coordinates point to somewhere inside the frame.
        # If not, set then to frame center.
        _x0 =  self._geometry.x0
        _y0 =  self._geometry.y0
        if _x0 is None or _x0 < 0 or _x0 >= self._image.shape[0] or \
           _y0 is None or _y0 < 0 or _y0 >= self._image.shape[1]:
            _x0 = self._image.shape[0] / 2
            _y0 = self._image.shape[1] / 2

        # 1/2 size of square window
        rwindow = len(self._in_mask) /2

        max_fom = 0.
        max_i = 0
        max_j = 0

        for i in range(int(_x0 - rwindow), int(_x0 + rwindow) + 1):
            for j in range(int(_y0 - rwindow), int(_y0 + rwindow) + 1):
                # Re-centering window.
                i1 = max(0, i - rwindow)
                j1 = max(0, j - rwindow)
                i2 = min(self._image.shape[0]-1, i + rwindow)
                j2 = min(self._image.shape[1]-1, j + rwindow)

                window = self._image[j1:j2,i1:i2]

                # averages in inner and outer regions.
                inner = ma.masked_array(window, mask=self._in_mask)
                outer = ma.masked_array(window, mask=self._out_mask)

                inner_sum = np.sum(inner) / self._in_mask_npix
                outer_sum = np.sum(outer) / self._out_mask_npix
                # inner_sum = np.sum(inner)
                # outer_sum = np.sum(outer)
                inner_std = np.std(inner)
                outer_std = np.std(outer)
                stddev = np.sqrt(inner_std**2 + outer_std**2)

                fom = (inner_sum - outer_sum) / stddev

                if fom > max_fom:
                    max_fom = fom
                    max_i = i
                    max_j = j

                print ('@@@@@@     line: 679  - ', fom, max_fom, i, j, " - ", i1, i2, j1, j2, " - ", inner_sum, outer_sum, (inner_sum-outer_sum))


        print ('@@@@@@     line: 698  - ', max_i, max_j)


# 68	        ic = 0
# 69	        jc = 0
# 70	        fom = 0.
# 71
# 72	        if (list) {
# 73	            call printf ("Running object locator... ")
# 74	            call flush (STDOUT)
# 75	        }
# 76
# 77	        # Scan window.
# 78	        do j = j1, j2 {
# 79	            do i = i1, i2 {
# 80
# 81	                # Extract two concentric circular samples.
# 82	                call el_get (im, sec, real(i), real(j), INNER_RADIUS, 0.0, 0.0,
# 83	                             bufx, bufy, nbuffer, NPOINT(is), NDATA(is),
# 84	                             mean1, std1, ASTEP(al), LINEAR(al),
# 85	                             INT_LINEAR, 4., 4.0, 0, SAREA(is))
# 86	                call el_get (im, sec, real(i), real(j), OUTER_RADIUS, 0.0, 0.0,
# 87	                             bufx, bufy, nbuffer, NPOINT(is), NDATA(is),
# 88	                             mean2, std2, ASTEP(al), LINEAR(al),
# 89	                             INT_LINEAR, 4., 4.0, 0, SAREA(is))
# 90
# 91	                # Figure of merit measures if there is reasonable
# 92	                # signal at position i,j.
# 93	                if (IS_INDEF(std1)) std1 = 0.0
# 94	                if (IS_INDEF(std2)) std2 = 0.0
# 95	                aux = std1 * std1 + std2 * std2
# 96
# 97
# 98
# 99	                if (aux > 0.0) {
# 100	                    aux = (mean1 - mean2) / sqrt(aux)
# 101	                    if (aux > fom) {
# 102	                        fom = aux
# 103	                        ic  = i
# 104	                        jc  = j
# 105	                    }
# 106	                }
# 107	            }
# 108	        }
# 109	        call mfree (bufy, TY_REAL)
# 110	        call mfree (bufx, TY_REAL)
# 111
# 112	        if (list)
# 113	            call printf ("Done.\n")
# 114
# 115	        # If valid object, re-center if asked for. Otherwise, no-detection
# 116	        # is signaled by setting center coordinates to INDEF.
# 117	        if (fom > thresh) {
# 118	            if (recenter) {
# 119	                XC(is) = real (ic)
# 120	                YC(is) = real (jc)
# 121	            }
# 122	        } else {
# 123	            XC(is) = INDEFR
# 124	            YC(is) = INDEFR
# 125	        }
# 126
# 127	        # Restore coordinates to physical system if needed.
# 128	        if ((!IS_INDEFR (XC(is))) && (!IS_INDEFR (YC(is)))) {
# 129	            if (PHYSICAL(is)) {
# 130	                XC(is) = el_s2p (im, XC(is), 1)
# 131	                YC(is) = el_s2p (im, YC(is), 2)
# 132	            }
# 133	        }
# 134
#
