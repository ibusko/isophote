from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

import unittest

import pyfits

from ellipse.sample import Sample
from ellipse.harmonics import fit_harmonics, harmonic_function


class TestEllipse(unittest.TestCase):

    def test_fitting(self):

        test_data = pyfits.open("synthetic_image.fits")
        test_data = test_data[0].data

        sample = Sample(test_data, 40., eps=0.4)
        s = sample.extract()

        y0, a1, b1, a2, b2 = fit_harmonics(s[0], s[1])

        # here we compute the gradient

        # and then compute the coefficient updates

