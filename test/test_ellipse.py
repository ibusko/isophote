from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

from util import build_test_data
from ellipse.sample import Sample
from ellipse.harmonics import fit_harmonics


class TestEllipse(unittest.TestCase):

    def test_gradient(self):

        test_data = build_test_data.build()

        sample = Sample(test_data, 40.)
        sample.update()

        self.assertAlmostEqual(sample.mean, 200.166,  3)
        self.assertAlmostEqual(sample.gradient, -4.178,  3)

    def test_fitting(self):

        test_data = build_test_data.build()

        # purposedly pick a first guess that is off in one of
        # the parameters. Ellipticity will affect b2.
        sample = Sample(test_data, 40., eps=0.3)
        s = sample.extract()

        y0, a1, b1, a2, b2 = fit_harmonics(s[0], s[2])

        print ('@@@@@@     line: 39  -  y0 ', y0)
        print ('@@@@@@     line: 38  -  a1 ', a1)
        print ('@@@@@@     line: 38  -  b1',  b1)
        print ('@@@@@@     line: 38  -  a2 ', a2)
        print ('@@@@@@     line: 38  -  b2 ', b2)


