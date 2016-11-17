from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

from ellipse.sample import Sample
from ellipse.harmonics import fit_harmonics, harmonic_function


class Fitter(object):

    def __init__(self, sample, conver=0.05, minit=10, maxit=50):
        self._sample = sample

        self._conver = conver
        self._minit = minit
        self._maxit = maxit

    def fit(self, crit=0.05):

        sample = self._sample

        for iter in range(self._maxit):

            sample.update()
            s = sample.extract()

            coeffs = fit_harmonics(s[0], s[2])

            # largest harmonic in absolute value drives the correction.
            largest_harmonic_index = np.argmax(np.abs(coeffs[1:]))
            largest_harmonic = coeffs[1:][largest_harmonic_index]

            # check if converged
            model = harmonic_function(s[0], coeffs[0], coeffs[1:])
            residual = s[2] - model
            # print ('@@@@@@     line: 40  - ', iter, np.std(residual), np.abs(largest_harmonic))
            if (crit * sample.sector_area * np.std(residual)) > np.abs(largest_harmonic):
                break

            # pick appropriate corrector code.
            corrector = correctors[largest_harmonic_index]

            # generate new Sample instance with corrected parameter
            sample = corrector.correct(sample, largest_harmonic)

        return sample


class ParameterCorrector(object):

    def correct(self, sample, harmonic):
        raise NotImplementedError


class EllipticityCorrector(ParameterCorrector):

    def correct(self, sample, harmonic):

        correction = harmonic * 2. * (1. - sample.geometry.eps) / sample.geometry.sma / sample.gradient

        new_eps = sample.geometry.eps - correction

        return Sample(sample.image, sample.geometry.sma,
                      x0 = sample.geometry.x0,
                      y0 = sample.geometry.y0,
                      astep = sample.geometry.astep,
                      eps = new_eps,
                      position_angle = sample.geometry.pa,
                      linear_growth = sample.geometry.linear_growth,
                      integrmode = sample.integrmode)


# instances of corrector code live here:

correctors = [EllipticityCorrector(),
              EllipticityCorrector(),
              EllipticityCorrector(),
              EllipticityCorrector()
]