from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import pyfits

from util import build_test_data
from ellipse.ellipse import Ellipse
from ellipse.isophote import Isophote


class TestEllipse(unittest.TestCase):

    def test_fit(self):

        # low noise image, fitted perfectly by sample.
        test_data = build_test_data.build()

        ellipse = Ellipse(test_data)
        isophote_list = ellipse.fit_image(maxsma=12.)

        self.assertIsInstance(isophote_list, list)
        self.assertGreater(len(isophote_list), 1)
        self.assertIsInstance(isophote_list[0], Isophote)

