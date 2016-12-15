from __future__ import (absolute_import, division, print_function, unicode_literals)

from ellipse.sample import Sample, CentralSample
from ellipse.fitter import Fitter, CentralFitter
from ellipse.isophote import Isophote

DEFAULT_STEP = 0.1


class Ellipse():
    '''
    This class provides the main access point to the isophote fitting algorithm.
    '''
    def __init__(self, image):
        '''
        Constructor

        :param image: np 2-D array
            image array
        '''
        self.image = image

    def fit_image(self, sma0=10., minsma=0., maxsma=100., step=DEFAULT_STEP, linear=False, maxrit=None):
        '''
        Main fitting method. Fits multiple isophotes on the image array passed
        to the constructor. This method basically loops over each one of the
        values of semi-major axis length (sma) constructed from the input parameters,
        and fits one isophote at each sma, returning the entire set of isophotes in
        a sorted list.

        :param sma0: float, default = 10.
            starting value for the semi-major axis length (pixels)
        :param minsma:  float, default = 0.
            minimum value for the semi-major axis length (pixels)
        :param maxsma: float, default=100.
            maximum value for the semi-major axis length (pixels)
        :param step: float, default = DEFAULT_STEP
            the step value being used to grow/shrink the semi-major
            axis length (pixels if 'linear=True', or relative value
            if 'linear=False')
        :param linear: boolean, default False
            semi-major axis growing/shrinking mode
        :param maxrit: float, default None
            maximum value of semi-major axis to perform an actual fit.
            Whenever the current semi-major axis length is larger than
            maxrit, the isophotes wil be just extracted using the current
            geometry, without being fitted. Ignored if None.
        :return: list
            this list stores fitted Isophote instances, sorted according
            to the semi-major axis length value.
        '''

        # multiple fitted isophotes will be stored here
        isophote_list = []

        # first, go from initial sma outwards until
        # hitting one of several stopping criteria.
        sma = sma0
        while True:
            isophote = self.fit_isophote(isophote_list, sma, step, linear, maxrit)

            # if abnormal condition, shut off iterative mode but keep going.
            if isophote.stop_code < 0:
                isophote.stop_code = 4
                if not maxrit:
                    maxrit = sma

            # figure out next sma; if exceeded user-defined
            # maximum, bail out from this loop.
            sma = isophote.sample.geometry.update_sma(step)
            if sma >= maxsma:
                break

        # reset sma so as to go inwards.
        first_isophote = isophote_list[0]
        sma, step = first_isophote.sample.geometry.reset_sma(step)

        # now, go from initial sma inwards towards center.
        while True:
            isophote = self.fit_isophote(isophote_list, sma, step, linear, maxrit)

            # figure out next sma; if exceeded user-defined
            # minimum, or too small, bail out from this loop
            sma = isophote.sample.geometry.update_sma(step)
            if sma <= max(minsma, 0.5):
                break

        # if user asked for minsma=0, extract special isophote there
        if minsma == 0.0:
            self.fit_isophote(isophote_list, 0.0, step, linear)

        # sort list of isophotes according to sma
        isophote_list.sort()

        return isophote_list

    def fit_isophote(self, isophote_list, sma, step=DEFAULT_STEP, linear=False, maxrit=None, noniterate=False):
        '''
        Fit one isophote with a given semi-major axis length.

        The 'step' and 'linear' parameters are not used to actually
        grow or shrink the current fitting semi-major axis length.
        They are necessary nevertheless, so the sampling algorithm
        can know where to start the gradient computation, and also
        how to compute the elliptical sector areas (when area
        integration mode is selected).

        :param isophote_list: list
            fitted Isophote instance is appended to this list. Must
            be created and managed by the caller.
        :param sma: float
            the semi-major axis length (pixels)
        :param step: float, default = DEFAULT_STEP
            the step value being used to grow/shrink the semi-major
            axis length (pixels)
        :param linear: boolean, default = False
            semi-major axis growing/shrinking mode
        :param maxrit: float, default None
            maximum value of semi-major axis to perform an actual fit.
            Whenever the current semi-major axis length is larger than
            maxrit, the isophote wil be just extracted using the current
            geometry, without fitting it. Ignored if None.
        :param noniterate: boolean, default False
            signals that the fitting algorithm should be bypassed and an
            isophote should be extracted with the geometry taken directly
            from the most recent Isophote instance stored in the
            'isophote_list' parameter.
            This parameter is mainly used when running the method in a loop
            over different values of semi-major axis length, and we want
            to change from iterative to non-iterative mode somewhere
            along the sequence of isophotes. When set to True, this
            parameter overrides the behavior associated with parameter
            'maxrit'.
        :return: Isophote instance
            the fitted isophote. The fitted isophote is also appended
            to the input list passed via parameter 'isophote_list'.
        '''
        # if available, geometry from last fitted isophote will be
        # used as initial guess for next isophote.
        geometry = None
        if len(isophote_list) > 0:
            geometry = isophote_list[-1].sample.geometry

        # do the fit.
        if noniterate or (maxrit and sma > maxrit):
            isophote = self._non_iterative(sma, step, linear, geometry)
        else:
            isophote = self._iterative(sma, step, linear, geometry)

        # store result in list, and report summary at stdout
        if isophote.valid:
            isophote_list.append(isophote)
            isophote.print()

        return isophote

    def _iterative(self, sma, step, linear, geometry):
        if sma > 0.:
            # iterative fitter
            sample = Sample(self.image, sma,
                            astep=step,
                            linear_growth=linear,
                            geometry=geometry)
            fitter = Fitter(sample)
        else:
            # sma == 0 requires special handling.
            sample = CentralSample(self.image, 0.0)
            fitter = CentralFitter(sample)

        isophote = fitter.fit()

        return isophote

    def _non_iterative(self, sma, step, linear, geometry):
        sample = Sample(self.image, sma,
                        astep=step,
                        linear_growth=linear,
                        geometry=geometry)
        sample.update()

        # build isophote without iterating with a Fitter
        isophote = Isophote(sample, 0, True, 4)

        return isophote




