from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np


class Isophote:

    def __init__(self, sample, iter, valid, stop_code):
        '''
        Container that helps in segregating information directly related to
        the sample (sampled intensities along an elliptical path on the image),
        from isophote-specific information.

        Parameters:
        ----------
        :param sample: instance of Sample
            the sample information
        :param iter: int
            number of iterations used to fit the isophote
        :param valid: boolean
            status of the fitting operation
        :param stop_code: int
            stop code:
                0 - normal.
                1 - less than pre-specified fraction of the extracted data
                    points are valid.
                2 - exceeded maximum number of iterations.
                3 - singular matrix in harmonic fit, results may not be valid.
                    Also signals insufficient number of data points to fit.
                4 - NOT IMPLEMENTED YET:
                    small or wrong gradient, or ellipse diverged; subsequent
                    ellipses at larger semi-major axis may have the same
                    constant geometric parameters.
                -1 - NOT IMPLEMENTED:
                     isophote was saved before completion of fit (by a cursor
                     command in interactive mode).

        Attributes:
        -----------
        :param intens: float
            mean intensity value along the elliptical path
        :param rms: float
            root-mean-sq of intensity values along the elliptical path
        :param int_err: float
            error of the mean (rms / sqrt(# data points)))
        :param pix_var: float
            estimate of pixel variance (rms * sqrt(average sector integration area))
        :param grad: float
            local radial intensity gradient
        '''
        self.sample = sample
        self.iter = iter
        self.valid = valid
        self.stop_code = stop_code

        self.intens = sample.mean

        self.rms = np.std(sample.values[2])
        self.int_err = self.rms / np.sqrt(sample.actual_points)
        self.pix_var = self.rms * np.sqrt(sample.sector_area)
        self.grad = sample.gradient

        self.tflux_e, self.tflux_c = self._compute_fluxes()

    def _compute_fluxes(self):
        # Compute integrated flux inside ellipse, as well as inside
        # circle defined by semi-major axis. Pixels in a square section
        # enclosing circle are scanned; the distance of each pixel to
        # the isophote center is compared both with the semi-major axis
        # length and with the length of the ellipse radius vector, and
        # integrals are updated if the pixel distance is smaller.

        # Compute limits of square array that encloses circle.
        sma = self.sample.geometry.sma
        x0 = self.sample.geometry.x0
        y0 = self.sample.geometry.y0
        xsize = self.sample.image.shape[1]
        ysize = self.sample.image.shape[0]

        imin = max(0, int(x0 - sma - 0.5) - 1)
        jmin = max(0, int(y0 - sma - 0.5) - 1)
        imax = min(xsize, int(x0 + sma + 0.5) + 1)
        jmax = min(ysize, int(y0 + sma + 0.5) + 1)

        # Integrate
        tflux_e = 0.
        tflux_c = 0.
        for j in range(jmin, jmax):
            for i in range(imin, imax):

                # radius of the circle and ellipse associated
                # with the given pixel.
                radius, angle = self.sample.geometry.to_polar(i, j)
                radius_e = self.sample.geometry.radius(angle)

                # pixel is inside circle with diameter given by sma
                if radius <= sma:
                    tflux_c += self.sample.image[j][i]

                # pixel is inside ellipse
                if radius <= radius_e:
                    tflux_e += self.sample.image[j][i]

        return tflux_e, tflux_c


