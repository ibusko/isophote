from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import numpy as np
import pyfits

import ellipse.integrator as I
from ellipse.sample import Sample, Geometry
from ellipse.integrator import NEAREST_NEIGHBOR, BI_LINEAR, MEAN


class TestGeometry(unittest.TestCase):

    def _check_geometry(self, geometry):

        sma1, sma2 = geometry._bounding_ellipses()

        self.assertAlmostEqual(sma1, 90.0, 3)
        self.assertAlmostEqual(sma2, 110.0, 3)

        # using an arbitrary angle of 0.5 rad. This is
        # to avoid a polar vector that sits on top of
        # one of the ellipse's axis.
        vertex_x, vertex_y = geometry.initialize_sector_geometry(0.5)

        self.assertAlmostEqual(geometry.sector_angular_width, 0.0571, 2)
        self.assertAlmostEqual(geometry.sector_area, 71.00, 2)

        self.assertAlmostEqual(vertex_x[0], 219.86, 2)
        self.assertAlmostEqual(vertex_x[1], 212.05, 2)
        self.assertAlmostEqual(vertex_x[2], 217.49, 2)
        self.assertAlmostEqual(vertex_x[3], 209.16, 2)

        self.assertAlmostEqual(vertex_y[0], 323.33, 2)
        self.assertAlmostEqual(vertex_y[1], 338.52, 2)
        self.assertAlmostEqual(vertex_y[2], 319.75, 2)
        self.assertAlmostEqual(vertex_y[3], 334.14, 2)

    def test_ellipse(self):

        # Geometrical steps
        geometry = Geometry(255., 255., 100., 0.4, np.pi/2, 0.2, False)

        self._check_geometry(geometry)

        # Linear steps
        geometry = Geometry(255., 255., 100., 0.4, np.pi/2, 20., True)

        self._check_geometry(geometry)

    def test_to_polar(self):
        # trivial case of a circle centered in (0.,0.)
        geometry = Geometry(0., 0., 100., 0.0, 0., 0.2, False)

        r, p = geometry.to_polar(100., 0.)
        self.assertAlmostEqual(r, 100., 2)
        self.assertAlmostEqual(p, 0., 4)

        r, p = geometry.to_polar(0., 100.)
        self.assertAlmostEqual(r, 100., 2)
        self.assertAlmostEqual(p, np.pi/2., 4)

        # vector with length 100. at 45 deg angle
        r, p = geometry.to_polar(70.71, 70.71)
        self.assertAlmostEqual(r, 100., 2)
        self.assertAlmostEqual(p, np.pi/4., 4)

        # position angle tilted 45 deg from X axis
        geometry = Geometry(0., 0., 100., 0.0, np.pi/4., 0.2, False)

        r, p = geometry.to_polar(100., 0.)
        self.assertAlmostEqual(r, 100., 2)
        self.assertAlmostEqual(p, np.pi*7./4., 4)

        r, p = geometry.to_polar(0., 100.)
        self.assertAlmostEqual(r, 100., 2)
        self.assertAlmostEqual(p, np.pi/4., 4)

        # vector with length 100. at 45 deg angle
        r, p = geometry.to_polar(70.71, 70.71)
        self.assertAlmostEqual(r, 100., 2)
        self.assertAlmostEqual(p, np.pi*2., 4)


class TestSampling(unittest.TestCase):

    def _init_test(self, integrmode=BI_LINEAR):
        test_data = pyfits.open("test_image.fits")
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

        self.assertEqual(len(s[0]), 13)
        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 0.1707,  3)
        self.assertAlmostEqual(np.std(s[2]),  0.00097, 3)
        # radii
        self.assertAlmostEqual(np.max(s[1]), 39.95, 2)
        self.assertAlmostEqual(np.min(s[1]), 24.70, 2)

