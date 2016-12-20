from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import numpy as np
from util import build_test_data

from ellipse.sample import Sample

from ellipse.integrator import NEAREST_NEIGHBOR, BI_LINEAR, MEAN, MEDIAN

test_data = None

class TestIntegrator(unittest.TestCase):

    def _init_test(self, integrmode=BI_LINEAR, sma=40., noise=1.E-6):

        global test_data
        if test_data is None:
            test_data = build_test_data.build(noise=noise)

        self.sample = Sample(test_data, sma, integrmode=integrmode)
        s = self.sample.extract()

        self.assertEqual(len(s), 3)
        self.assertEqual(len(s[0]), len(s[1]))
        self.assertEqual(len(s[0]), len(s[2]))

        return s

    def test_bilinear(self):

        s = self._init_test()

        self.assertEqual(len(s[0]), 223)

        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 200.018, 3)
        self.assertAlmostEqual(np.std(s[2]),  0.0159, 4)

        # radii
        self.assertAlmostEqual(np.max(s[1]), 40.0, 2)
        self.assertAlmostEqual(np.min(s[1]), 32.0, 2)

        self.assertEqual(self.sample.total_points, 223)
        self.assertEqual(self.sample.actual_points, 223)

    def test_bilinear_small(self):

        s = self._init_test(sma=10., noise=1.E-12)

        # self.assertEqual(len(s[0]), 28)

        for k in range(len(s[2])):
            print("@@@@@@  file test_integrator.py; line 55 - ",  k, "  " , s[2][k])
        print("@@@@@@  file test_integrator.py; line 56 - ",  np.std(s[2]))

        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 2347.4, 1)
        self.assertAlmostEqual(np.std(s[2]),  80.9, 1)

        # radii
        self.assertAlmostEqual(np.max(s[1]), 40.0, 2)
        self.assertAlmostEqual(np.min(s[1]), 32.0, 2)

        self.assertEqual(self.sample.total_points, 223)
        self.assertEqual(self.sample.actual_points, 223)

    def test_nearest_neighbor(self):

        s = self._init_test(integrmode=NEAREST_NEIGHBOR)

        self.assertEqual(len(s[0]), 112)
        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 200.434,  3)
        self.assertAlmostEqual(np.std(s[2]),  3.132, 3)
        # radii
        self.assertAlmostEqual(np.max(s[1]), 40.0, 2)
        self.assertAlmostEqual(np.min(s[1]), 32.0, 2)

        self.assertEqual(self.sample.total_points, 112)
        self.assertEqual(self.sample.actual_points, 112)

    def test_mean(self):

        s = self._init_test(integrmode=MEAN)

        self.assertEqual(len(s[0]), 37)
        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 200.575,  3)
        self.assertAlmostEqual(np.std(s[2]),  1.339, 3)
        # radii
        self.assertAlmostEqual(np.max(s[1]), 39.98, 2)
        self.assertAlmostEqual(np.min(s[1]), 32.028, 2)

        self.assertAlmostEqual(self.sample.sector_area, 21.4, 1)
        self.assertEqual(self.sample.total_points, 37)
        self.assertEqual(self.sample.actual_points, 37)

    def test_median(self):

        s = self._init_test(integrmode=MEDIAN)

        self.assertEqual(len(s[0]), 37)
        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 200.730,  3)
        self.assertAlmostEqual(np.std(s[2]),  1.581, 3)
        # radii
        self.assertAlmostEqual(np.max(s[1]), 39.98, 2)
        self.assertAlmostEqual(np.min(s[1]), 32.028, 2)

        self.assertAlmostEqual(self.sample.sector_area, 21.4, 1)
        self.assertEqual(self.sample.total_points, 37)
        self.assertEqual(self.sample.actual_points, 37)
