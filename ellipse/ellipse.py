from __future__ import (absolute_import, division, print_function, unicode_literals)

from ellipse.sample import Sample, CentralSample
from ellipse.fitter import Fitter, CentralFitter
from ellipse.isophote import Isophote

DEFAULT_STEP = 0.1


class Ellipse():

            # :param isophote_list: list
            # list that gets appended with a new, fitted Isophote instance.
            # Must be initialized by the caller.


    def __init__(self, image):
        self.image = image

    def fit_image(self, sma0=10., minsma=0., maxsma=100., step=DEFAULT_STEP, linear=False, maxrit=None):
        self.isophote_list = []

        # first, go from initial sma outwards until
        # hitting one of several stopping criteria.
        sma = sma0
        while True:
            isophote = self.fit_isophote(sma, step, linear, maxrit)

            # figure out next sma; if too large, bail out
            sma = isophote.sample.geometry.update_sma(step)
            if sma >= maxsma:
                break

        # reset sma so as to go inwards.
        first_isophote = self.isophote_list[0]
        sma, step = first_isophote.sample.geometry.reset_sma(step)

        # now, go from initial sma inwards
        while True:
            isophote = self.fit_isophote(sma, step, linear, maxrit)

            # figure out next sma; if too small, bail out
            sma = isophote.sample.geometry.update_sma(step)
            if sma <= max(minsma, 0.75):
                break

        # if user asked for minsma=0, extract special isophote there
        if minsma == 0.0:
            isophote = self.fit_isophote(0.0, step, linear)

        # sort list of isophotes according to sma
        self.isophote_list.sort()

        return self.isophote_list

    def fit_isophote(self, sma, step=DEFAULT_STEP, linear=False, maxrit=None):
        '''
        Fit an isophote with a given semi-major axis length.

        The 'step' and 'linear' parameters are not used to actually
        grow or shrink the current fitting semi-major axis length.
        They are necessary nevertheless, so the sampling algorithm
        can know where to start the gradient computation, and also
        how to compute the elliptical sector areas (when area
        integration mode is selected).

        :param sma: float
            the semi-major axis length (pixels)
        :param step: float
            the step value being used to grow/shrink the semi-major
            axis length (pixels)
        :param linear: boolean
            semi-major axis growing/shrinking mode
        :param maxrit: float
            maximum value of semi-major axis to perform an actual fit.
            If sma>maxrit, just extract the sample at current geometry,
            without fitting it.
        :return: Isophote instance
            the fitted isophote. The fitted isophote is also appended
            to class attribute 'isophote_list'
        '''
        if maxrit and sma > maxrit:
            # build isophote in non-iterative mode.
            sample = Sample(self.image, sma, geometry=self.isophote_list[-1].sample.geometry)
            sample.update()

            isophote = Isophote(sample, 0, True, 4)
            self.isophote_list.append(isophote)
            isophote.print()

            return isophote
        else:
            if sma > 0.:
                sample = Sample(self.image, sma, astep=step, linear_growth=linear)
                fitter = Fitter(sample)
            else:
                sample = CentralSample(self.image, 0.0)
                fitter = CentralFitter(sample)

            isophote = fitter.fit()

            if isophote.valid:
                self.isophote_list.append(isophote)
                isophote.print()

            return isophote

