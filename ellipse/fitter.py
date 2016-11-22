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

        values = None

        for iter in range(maxit):

            # Force the sample to compute its gradient and associated values.
            sample.update()

            # The extract() method returns sampled values as a 2-d numpy array
            # with the following structure:
            # values[0] = 1-d array with angles
            # values[1] = 1-d array with radii
            # values[2] = 1-d array with intensity
            values = sample.extract()

            # Fit harmonic coefficients. Failure in fitting is
            # a fatal error; terminate immediately with sample
            # marked as invalid.
            try:
                coeffs = fit_harmonics(values[0], values[2])
            except RuntimeError as e:
                print(e)
                self._update_sample(sample, values, iter+1, False)
                return sample

            # largest harmonic in absolute value drives the correction.
            largest_harmonic_index = np.argmax(np.abs(coeffs[1:]))
            largest_harmonic = coeffs[1:][largest_harmonic_index]

            # check if converged
            model = harmonic_function(values[0], coeffs[0], coeffs[1:])
            residual = values[2] - model
            # print ('@@@@@@     line: 40  - ', iter, np.std(residual), np.abs(largest_harmonic),
            #        largest_harmonic_index, sample.geometry.x0)
            if (conver * sample.sector_area * np.std(residual)) > np.abs(largest_harmonic):
                # Got a valid solution. But before returning, ensure
                # that a minimum of iterations has run.
                if iter >= minit:
                    self._update_sample(sample, values, sample.mean, sample.gradient, iter+1, True)
                    return sample

            # pick appropriate corrector code.
            corrector = correctors[largest_harmonic_index]

            # generate *NEW* Sample instance with corrected parameter. Note that
            # this instance is still empty of other information besides its geometry.
            # It needs to be explicitly updated in case we need to return it as the
            # result of the fit operation.
            # We have to build a new Sample instance every time because of the lazy
            # extraction process used by Sample code. To minimize the number of
            # calls to the area integrators, we pay a (hopefully smaller) price here,
            # by having multiple calls to the Sample constructor.
            #TODO
            # which begs the question: should we then return an object of a different
            # class? Such as an Isophote class, which will allow better segregation of
            # attributes associated with a sample (mean, gradient, values, etc) from
            # attributes associated with the actual fitting (iter, valid flag, etc)
            sample = corrector.correct(sample, largest_harmonic)

        # even when running out of iterations, we consider the sample as
        # valid. Not sure if this is 100% correct. We'll see as we proceed
        # with adding more termination criteria.
        self._update_sample(sample, values, self._sample.mean, self._sample.gradient, maxit, True)

        return sample

    def _update_sample(self, sample, values, mean, gradient, iter, valid):
        # this is required because we build a new instance of Sample at
        # every single iteration step. Modifying an existing Sample instead
        # proved to result in more convoluted code with worse encapsulation,
        # and hard-to-understand logic related to the caching used in sample
        # extraction.
        sample.iter = iter
        sample.valid = valid
        sample.values = values
        sample.mean = mean
        sample.gradient = gradient


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


