from __future__ import (absolute_import, division, print_function, unicode_literals)

import math
import numpy as np

from ellipse.geometry import normalize_angle
from ellipse.harmonics import fit_1st_and_2nd_harmonics, first_and_2nd_harmonic_function
from ellipse.sample import Sample, sample_copy
from ellipse.isophote import Isophote, CentralPixel


class Fitter(object):
    '''
    The main fitter class.

    '''
    def __init__(self, sample):
        '''
        Create a Fitter instance for a given Sample instance.

        :param sample: instance of Sample
            the sample to be fitted
        '''
        self._sample = sample

    def fit(self, conver=0.05, minit=10, maxit=50, fflag=0.5):
        '''
        Perform the actual fit, returning an Isophote instance:

            fitter = Fitter(sample)
            isophote = fitter.fit()


        :param conver: float
            main convergency criterion. Largest harmonic amplitude
            must be smaller than 'conver' times the fit rms.
        :param minit: int
            minimum number of iterations to perform
        :param maxit: int
            maximum number of iterations to perform
        :param fflag: float
            acceptable fraction of flagged data points in sample.
            If the actual number of valid data points is smaller
            than this, stop iterating and return current Isophote.
            For now, flagged data points are points that lie outside
            the image frame. In the future, it may include masked
            pixels as well.
        :return: instance of Isophote
            isophote with the fitted sample plus additional fit status
            information
        '''
        sample = self._sample

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
                coeffs = fit_1st_and_2nd_harmonics(values[0], values[2])
            except Exception as e:
                print(e)
                sample_copy(self._sample, sample)
                return Isophote(sample, iter+1, False, 3)

            # largest harmonic in absolute value drives the correction.
            largest_harmonic_index = np.argmax(np.abs(coeffs[1:]))
            largest_harmonic = coeffs[1:][largest_harmonic_index]

            # check if converged
            model = first_and_2nd_harmonic_function(values[0], coeffs)
            residual = values[2] - model

            if (conver * sample.sector_area * np.std(residual)) > np.abs(largest_harmonic):
                # Got a valid solution. But before returning, ensure
                # that a minimum of iterations has run.
                if iter >= minit:
                    # This copy of one sample into another is required because we build
                    # a new instance of Sample at every single iteration step. Modifying
                    # the same Sample every time instead proved to result in more
                    # convoluted code with worse encapsulation, and hard-to-understand
                    # logic related to the caching used in sample extraction.
                    sample_copy(self._sample, sample)
                    return Isophote(sample, iter+1, True, 0)

            # it may not have converged yet, but the sample contains too
            # many invalid data points: return.
            if sample.actual_points < (sample.total_points * (1. - fflag)):
                sample_copy(self._sample, sample)
                return Isophote(sample, iter+1, True, 1)

            # pick appropriate corrector code.
            corrector = _correctors[largest_harmonic_index]

            # generate *NEW* Sample instance with corrected parameter. Note that
            # this instance is still devoid of other information besides its geometry.
            # It needs to be explicitly updated in case we need to return it as the
            # result of the fit operation.
            # We have to build a new Sample instance every time because of the lazy
            # extraction process used by Sample code. To minimize the number of
            # calls to the area integrators, we pay a (hopefully smaller) price here,
            # by having multiple calls to the Sample constructor.
            sample = corrector.correct(sample, largest_harmonic)

        # even when running out of iterations, we consider the isophote as
        # valid. Not sure if this is 100% correct. We'll see as we proceed
        # with adding more termination criteria. Copy sample info before
        # returning!
        sample_copy(self._sample, sample)
        return Isophote(sample, maxit, True, 2)


class _ParameterCorrector(object):

    def correct(self, sample, harmonic):
        raise NotImplementedError


class _PositionCorrector(_ParameterCorrector):

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

class _PositionCorrector_0(_PositionCorrector):

    def correct(self, sample, harmonic):

        aux = -harmonic * (1. - sample.geometry.eps) / sample.gradient

        dx = -aux * math.sin(sample.geometry.pa)
        dy =  aux * math.cos(sample.geometry.pa)

        return self.finalize_correction(dx, dy, sample)


class _PositionCorrector_1(_PositionCorrector):

    def correct(self, sample, harmonic):

        aux = -harmonic / sample.gradient

        dx = aux * math.cos(sample.geometry.pa)
        dy = aux * math.sin(sample.geometry.pa)

        return self.finalize_correction(dx, dy, sample)


class _AngleCorrector(_ParameterCorrector):

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


class _EllipticityCorrector(_ParameterCorrector):

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

_correctors = [_PositionCorrector_0(),
               _PositionCorrector_1(),
               _AngleCorrector(),
               _EllipticityCorrector()
]


class CentralFitter(Fitter):
    '''
    Derived Fitter class, designed specifically to handle the
    case of the central pixel in the galaxy image.
    '''
    def fit(self):
        '''
        Overrides the base class to perform just a simple 1-pixel
        extraction at the current x0,y0 position, using bi-linear
        interpolation.

        :return: instance of the CentralPixel class.
            For convenience, the CentralPixel class inherits from
            the Isophote class, although it's not really a true
            isophote but just a single intensity value at the central
            position. Thus, most of its attributes are hardcoded to
            None, or other default value when appropriate.
        '''
        self._sample.update()
        return CentralPixel(self._sample)



