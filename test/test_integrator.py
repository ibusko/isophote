from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import numpy as np
import pyfits

from ellipse.sample import Sample

from ellipse.integrator import NEAREST_NEIGHBOR, BI_LINEAR, MEAN


class TestIntegrator(unittest.TestCase):

    def _init_test(self, integrmode=BI_LINEAR):
        test_data = pyfits.open("synthetic_image.fits")
        test_data = test_data[0].data

        sample = Sample(test_data, 40., eps=0.4, integrmode=integrmode)
        s = sample.extract()

        self.assertEqual(len(s), 3)
        self.assertEqual(len(s[0]), len(s[1]))
        self.assertEqual(len(s[0]), len(s[2]))

        return s

    def test_bilinear(self):

        s = self._init_test()

        self.assertEqual(len(s[0]), 191)
        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 0.1717, 3)
        self.assertAlmostEqual(np.std(s[2]),  0.0022, 3)

        # radii
        self.assertAlmostEqual(np.max(s[1]), 40.0, 2)
        self.assertAlmostEqual(np.min(s[1]), 24.0, 2)

    def test_nearest_neighbor(self):

        s = self._init_test(integrmode=NEAREST_NEIGHBOR)

        self.assertEqual(len(s[0]), 96)
        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 0.1717,  3)
        self.assertAlmostEqual(np.std(s[2]),  0.00097, 3)
        # radii
        self.assertAlmostEqual(np.max(s[1]), 40.0, 2)
        self.assertAlmostEqual(np.min(s[1]), 24.001, 2)

    def test_mean(self):

        s = self._init_test(integrmode=MEAN)

        self.assertEqual(len(s[0]), 38)
        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 0.1716,  3)
        self.assertAlmostEqual(np.std(s[2]),  0.001593, 3)
        # radii
        self.assertAlmostEqual(np.max(s[1]), 39.95, 2)
        self.assertAlmostEqual(np.min(s[1]), 24.03, 2)

