from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import numpy as np
import pyfits

import ellipse.integrator as I
from ellipse.sample import Sample, Geometry
from ellipse.integrator import NEAREST_NEIGHBOR


class TestGeometry(unittest.TestCase):

    def test_ellipse(self):

        geometry = Geometry(255., 255., 100., 0.4, np.pi/2)

        a1, a2 = I.limiting_ellipses(geometry.sma, 0.2, False)

        self.assertAlmostEqual(a1, 90.0,  3)
        self.assertAlmostEqual(a2, 110.0, 3)

        a1, a2 = I.limiting_ellipses(geometry.sma, 20., True)

        self.assertAlmostEqual(a1, 90.0,  3)
        self.assertAlmostEqual(a2, 110.0, 3)


class TestSampling(unittest.TestCase):

    def test_bilinear(self):

        test_data = pyfits.open("test_image.fits")
        test_data = test_data[0].data

        sample = Sample(test_data, 40., eps=0.4)
        s = sample.extract()

        self.assertEqual(len(s), 3)
        self.assertEqual(len(s[0]), len(s[1]))
        self.assertEqual(len(s[0]), len(s[2]))

        # values for image test_image.fits, sma=40., eps=0.4
        self.assertEqual(len(s[0]), 191)
        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 0.1717, 3)
        self.assertAlmostEqual(np.std(s[2]),  0.0022, 3)

        # radii
        self.assertAlmostEqual(np.max(s[1]), 40.0, 2)
        self.assertAlmostEqual(np.min(s[1]), 24.0, 2)

    def test_nearest_neighbor(self):

        test_data = pyfits.open("test_image.fits")
        test_data = test_data[0].data

        sample = Sample(test_data, 40., eps=0.4, integrmode=NEAREST_NEIGHBOR)
        s = sample.extract()

        self.assertEqual(len(s), 3)
        self.assertEqual(len(s[0]), len(s[1]))
        self.assertEqual(len(s[0]), len(s[2]))

        # values for image test_image.fits, sma=40., eps=0.4
        self.assertEqual(len(s[0]), 96)
        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 0.1717,  3)
        self.assertAlmostEqual(np.std(s[2]),  0.00097, 3)
        # radii
        self.assertAlmostEqual(np.max(s[1]), 40.0, 2)
        self.assertAlmostEqual(np.min(s[1]), 24.001, 2)

