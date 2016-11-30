from __future__ import (absolute_import, division, print_function, unicode_literals)

from ellipse.sample import Sample
from ellipse.fitter import Fitter


class Ellipse():

    def __init__(self, image):
        self.image = image

    def fit(self, sma0=10., minsma=0., maxsma=100., step=0.1, linear=False):

        sma = sma0
        result = []
        go = True

        while go:
            sample = Sample(self.image, sma)
            fitter = Fitter(sample)
            iso = fitter.fit()

            result.append(iso)

            if linear:
                sma += step
            else:
                sma *= (1. + step)

            if sma >= maxsma or not iso.valid:
                break

        return result
