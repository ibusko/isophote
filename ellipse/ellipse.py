from __future__ import (absolute_import, division, print_function, unicode_literals)

from ellipse.sample import Sample, CentralSample
from ellipse.fitter import Fitter, CentralFitter


class Ellipse():

    def __init__(self, image):
        self.image = image

    def fit_image(self, sma0=10., minsma=0., maxsma=100., step=0.1, linear=False):
        result = []

        # first, go from initial sma outwards until
        # hitting one of several stopping criteria.
        sma = sma0
        while True:
            isophote = self.fit_isophote(result, sma, step, linear)

            if sma >= maxsma:
                break

            if linear:
                sma += step
            else:
                sma *= (1. + step)

        # now reset sma to go inwards.
        if linear:
            sma = sma0 - step
        else:
            sma = sma0 / (1. + step)

        # finally, go from initial sma inwards
        while True:
            isophote = self.fit_isophote(result, sma, step, linear)

            # stop going inwards when sma becomes too small.
            if sma <= max(minsma, 0.75):
                break

            if linear:
                sma -= step
            else:
                sma /= (1. + step)

        # if user asked for minsma=0, extract special isophote there
        if minsma == 0.0:
            isophote = self.fit_isophote(result, 0.0, step, linear)

        return result

    def fit_isophote(self, isophote_list, sma, step, linear):
        '''
        Fit an isophote with a given semi-major axis length.

        The 'step' and 'linear' parameters are not used to actually
        grow or shrink the current fitting semi-major axis length.
        They are necessary nevertheless, so the sampling algorithm
        can know where to start the gradient computation, and also
        how to compute the elliptical sector areas (when area
        integration mode is selected).

        :param isophote_list: list
            list that gets appended with the resulting Isophote instance.
            Must be initialized by the caller.
        :param sma: float
            the semi-major axis length (pixels)
        :param step: float
            the semi-major axis length (pixels)
        :param linear: boolean
            semi-major axis growing/shrinking mode
        :return: Isophote instance
            the fitted isophote
        '''
        if sma > 0.:
            sample = Sample(self.image, sma, astep=step, linear_growth=linear)
            fitter = Fitter(sample)
        else:
            sample = CentralSample(self.image, 0.0)
            fitter = CentralFitter(sample)

        isophote = fitter.fit()

        if isophote.valid:
            isophote_list.append(isophote)

            print ('@@@@@@     line: 79  - ',isophote.sample.geometry.sma, isophote.intens)

        return isophote

