from __future__ import (absolute_import, division, print_function, unicode_literals)

import math
import numpy as np

from ellipse.geometry import normalize_angle
from ellipse.harmonics import fit_harmonics, harmonic_function
from ellipse.sample import Sample


class Fitter(object):

    def __init__(self, sample):
        self._sample = sample

    def fit(self, conver=0.05, minit=10, maxit=50):

        sample = self._sample

        # make sure sample status indicates
        # that no valid fit was gotten yet.
        sample.iter = None
        sample.valid = False

        for iter in range(maxit):

            # Force the sample to compute its gradient and associated values.
            sample.update()

            # iter is always reported in the Sample instance,
            # regardless of the validity status.
            sample.iter = iter

            s = sample.extract()

            # Fit harmonic coefficients. Failure in fitting is
            # a fatal error; terminate immediately with sample
            # marked as invalid.
            try:
                coeffs = fit_harmonics(s[0], s[2])
            except RuntimeError as e:
                print(e)
                return sample

            # largest harmonic in absolute value drives the correction.
            largest_harmonic_index = np.argmax(np.abs(coeffs[1:]))
            largest_harmonic = coeffs[1:][largest_harmonic_index]

            # check if converged
            model = harmonic_function(s[0], coeffs[0], coeffs[1:])
            residual = s[2] - model
            # print ('@@@@@@     line: 40  - ', iter, np.std(residual), np.abs(largest_harmonic),
            #        largest_harmonic_index, sample.geometry.x0)
            if (conver * sample.sector_area * np.std(residual)) > np.abs(largest_harmonic):
                # Got a valid solution. But before returning, ensure
                # that a minimum of iterations has run.
                if iter >= minit:
                    sample.valid = True
                    return sample

            # pick appropriate corrector code.
            corrector = correctors[largest_harmonic_index]

            # generate new Sample instance with corrected parameter
            sample = corrector.correct(sample, largest_harmonic)

        # even when running out of iterations, we consider the sample as
        # valid. Not sure if this is 100% correct. We'll see as we proceed
        # with adding more termination criteria.
        sample.valid = True

        return sample


class ParameterCorrector(object):

    def correct(self, sample, harmonic):
        raise NotImplementedError


class PositionCorrector(ParameterCorrector):

    def finalize_correction(self, dx, dy, sample):
        new_x0 = sample.geometry.x0 + dx
        new_y0 = sample.geometry.y0 + dy
        return Sample(sample.image, sample.geometry.sma,
                      x0=new_x0,
                      y0=new_y0,
                      astep=sample.geometry.astep,
                      eps=sample.geometry.eps,
                      position_angle=sample.geometry.pa,
                      linear_growth=sample.geometry.linear_growth,
                      integrmode=sample.integrmode)

class PositionCorrector_0(PositionCorrector):

    def correct(self, sample, harmonic):

        aux = -harmonic * (1. - sample.geometry.eps) / sample.gradient

        dx = -aux * math.sin(sample.geometry.pa)
        dy =  aux * math.cos(sample.geometry.pa)

        return self.finalize_correction(dx, dy, sample)


class PositionCorrector_1(PositionCorrector):

    def correct(self, sample, harmonic):

        aux = -harmonic / sample.gradient

        dx = aux * math.cos(sample.geometry.pa)
        dy = aux * math.sin(sample.geometry.pa)

        return self.finalize_correction(dx, dy, sample)


class AngleCorrector(ParameterCorrector):

    def correct(self, sample, harmonic):

        eps = sample.geometry.eps
        sma = sample.geometry.sma
        gradient = sample.gradient

        correction = harmonic * 2. * (1. - eps) /  sma / gradient / ((1. - eps)**2 - 1.)

        new_pa = normalize_angle(sample.geometry.pa + correction)

        return Sample(sample.image, sample.geometry.sma,
                      x0 = sample.geometry.x0,
                      y0 = sample.geometry.y0,
                      astep = sample.geometry.astep,
                      eps = sample.geometry.eps,
                      position_angle = new_pa,
                      linear_growth = sample.geometry.linear_growth,
                      integrmode = sample.integrmode)


class EllipticityCorrector(ParameterCorrector):

    def correct(self, sample, harmonic):

        eps = sample.geometry.eps
        sma = sample.geometry.sma
        gradient = sample.gradient

        correction = harmonic * 2. * (1. - eps) / sma / gradient

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

correctors = [PositionCorrector_0(),
              PositionCorrector_1(),
              AngleCorrector(),
              EllipticityCorrector()
]


