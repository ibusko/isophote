from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

import unittest

import pyfits

from ellipse.sample import Sample
from ellipse.harmonics import fit_harmonics, harmonic_function
from ellipse.integrator import NEAREST_NEIGHBOR


class TestEllipse(unittest.TestCase):

    def _get_sample(self, eps=0.4):
        # test_data = pyfits.open("synthetic_image_2.fits")
        test_data = pyfits.open("M51.fits")
        test_data = test_data[0].data

        sample = Sample(test_data, 40., eps=eps, integrmode=NEAREST_NEIGHBOR)
        sample.update()

        return sample

    def test_gradient(self):

        sample = self._get_sample()

        print ('@@@@@@     line: 30  - ', sample.mean)
        print ('@@@@@@     line: 31  - ', sample.gradient)

        self.assertAlmostEqual(sample.mean, 103.131,  3)
        self.assertAlmostEqual(sample.gradient, -0.000688,  3)

    # def test_fitting(self):
    #
    #     sample = self._get_sample()
    #     s = sample.extract()
    #
    #     y0, a1, b1, a2, b2 = fit_harmonics(s[0], s[1])
    #
    #     print ('@@@@@@     line: 39  -  y0 ', y0)
    #     print ('@@@@@@     line: 38  -  a1 ', a1)
    #     print ('@@@@@@     line: 38  -  b1',  b1)
    #     print ('@@@@@@     line: 38  -  a2 ', a2)
    #     print ('@@@@@@     line: 38  -  b2 ', b2)


