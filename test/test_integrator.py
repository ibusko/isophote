from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import numpy as np
from util import build_test_data

from ellipse.sample import Sample

from ellipse.integrator import NEAREST_NEIGHBOR, BI_LINEAR, MEAN, MEDIAN

test_data = None

class TestIntegrator(unittest.TestCase):

    def _init_test(self, integrmode=BI_LINEAR):

        global test_data
        if test_data is None:
            test_data = build_test_data.build()

        sample = Sample(test_data, 40., integrmode=integrmode)
        s = sample.extract()

        self.assertEqual(len(s), 3)
        self.assertEqual(len(s[0]), len(s[1]))
        self.assertEqual(len(s[0]), len(s[2]))

        return s

    def test_bilinear(self):

        s = self._init_test()

        self.assertEqual(len(s[0]), 223)
        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 200.166, 3)
        self.assertAlmostEqual(np.std(s[2]),  2.073, 3)

        # radii
        self.assertAlmostEqual(np.max(s[1]), 40.0, 2)
        self.assertAlmostEqual(np.min(s[1]), 32.0, 2)

    def test_nearest_neighbor(self):

        s = self._init_test(integrmode=NEAREST_NEIGHBOR)

        self.assertEqual(len(s[0]), 112)
        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 200.434,  3)
        self.assertAlmostEqual(np.std(s[2]),  3.132, 3)
        # radii
        self.assertAlmostEqual(np.max(s[1]), 40.0, 2)
        self.assertAlmostEqual(np.min(s[1]), 32.0, 2)

    def test_mean(self):

        s = self._init_test(integrmode=MEAN)

        self.assertEqual(len(s[0]), 37)
        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 200.575,  3)
        self.assertAlmostEqual(np.std(s[2]),  1.339, 3)
        # radii
        self.assertAlmostEqual(np.max(s[1]), 39.98, 2)
        self.assertAlmostEqual(np.min(s[1]), 32.028, 2)

    def test_median(self):

        s = self._init_test(integrmode=MEDIAN)

        self.assertEqual(len(s[0]), 37)
        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 200.730,  3)
        self.assertAlmostEqual(np.std(s[2]),  1.581, 3)
        # radii
        self.assertAlmostEqual(np.max(s[1]), 39.98, 2)
        self.assertAlmostEqual(np.min(s[1]), 32.028, 2)

