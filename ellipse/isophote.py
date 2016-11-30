from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

from ellipse.harmonics import fit_upper_harmonic



class Isophote:

    def __init__(self, sample, niter, valid, stop_code):
        '''
        Container that helps in segregating information directly related to
        the sample (sampled intensities along an elliptical path on the image),
        from isophote-specific information.

        Parameters:
        ----------
        :param sample: instance of Sample
            the sample information
        :param niter: int
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
        :param grad_error: float
            measurement error of local radial intensity gradient
        :param grad_r_error: float
            relative error of local radial intensity gradient
        :param npix_e: int
            total number of valid pixels inside ellipse
        :param npix_c: int
            total number of valid pixels inside circle
        :param sarea: float
            average sector area on isophote (pixel)
        :param ndata: int
            number of valid data points on isophote
        :param nflag: int
            number of flagged data points on isophote
        '''
        self.sample = sample
        self.niter = niter
        self.valid = valid
        self.stop_code = stop_code

        self.intens = sample.mean

        self.rms = np.std(sample.values[2])
        self.int_err = self.rms / np.sqrt(sample.actual_points)
        self.pix_var = self.rms * np.sqrt(sample.sector_area)
        self.grad = sample.gradient
        self.grad_error = sample.gradient_error
        self.grad_r_error = sample.gradient_relative_error
        self.sarea = sample.sector_area
        self.ndata = sample.actual_points
        self.nflag = sample.total_points - sample.actual_points

        # flux contained inside ellipse and circle
        self.tflux_e, self.tflux_c, self.npix_e, self.npix_c = self._compute_fluxes()

        # deviations from perfect ellipticity
        c = fit_upper_harmonic(sample.values[0], sample.values[2], 3)
        self.a3 = c[1] / sample.geometry.sma /sample.gradient
        self.b3 = c[2] / sample.geometry.sma /sample.gradient
        c = fit_upper_harmonic(sample.values[0], sample.values[2], 4)
        self.a4 = c[1] / sample.geometry.sma /sample.gradient
        self.b4 = c[2] / sample.geometry.sma /sample.gradient

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
        npix_e = 0
        npix_c = 0
        for j in range(jmin, jmax):
            for i in range(imin, imax):

                # radius of the circle and ellipse associated
                # with the given pixel.
                radius, angle = self.sample.geometry.to_polar(i, j)
                radius_e = self.sample.geometry.radius(angle)

                # pixel is inside circle with diameter given by sma
                if radius <= sma:
                    tflux_c += self.sample.image[j][i]
                    npix_c += 1

                # pixel is inside ellipse
                if radius <= radius_e:
                    tflux_e += self.sample.image[j][i]
                    npix_e += 1

        return tflux_e, tflux_c, npix_e, npix_c


