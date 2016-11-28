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
