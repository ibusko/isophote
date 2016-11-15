from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

import unittest

import pyfits

from ellipse.sample import Sample
from ellipse.harmonics import fit_harmonics, harmonic_function


class TestEllipse(unittest.TestCase):

    def _get_sample(self):
        test_data = pyfits.open("synthetic_image.fits")
        test_data = test_data[0].data

        sample = Sample(test_data, 40., eps=0.4)
        sample.update()

        return sample

    def test_gradient(self):

        sample = self._get_sample()

        self.assertAlmostEqual(sample.mean, 31.465,  3)
        self.assertAlmostEqual(sample.gradient, -0.786,  3)

    def test_fitting(self):

        sample = self._get_sample()

        s = sample.extract()

        y0, a1, b1, a2, b2 = fit_harmonics(s[0], s[1])

        # and then compute the coefficient updates

