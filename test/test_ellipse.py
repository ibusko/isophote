from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import numpy as np
import pyfits

from util import build_test_data
from ellipse.integrator import MEAN
from ellipse.ellipse import Ellipse
from ellipse.isophote import Isophote


class TestEllipse(unittest.TestCase):

    def test_fit(self):

        # low noise image, fitted perfectly by sample.
        test_data = build_test_data.build(pa=10./180.*np.pi, noise=1.E-12)

        ellipse = Ellipse(test_data)
        isophote_list = ellipse.fit_image()

        self.assertIsInstance(isophote_list, list)
        self.assertGreater(len(isophote_list), 1)
        self.assertIsInstance(isophote_list[0], Isophote)

        # verify that the list is properly sorted in sem-major axis length
        self.assertGreater(isophote_list[-1], isophote_list[0])

    def test_m51(self):
        image = pyfits.open("data/M51.fits")
        test_data = image[0].data
        ellipse = Ellipse(test_data)
        isophote_list = ellipse.fit_image()

