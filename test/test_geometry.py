from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import numpy as np

from ellipse.geometry import Geometry


class TestGeometry(unittest.TestCase):

    def _check_geometry(self, geometry):

        sma1, sma2 = geometry.bounding_ellipses()

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

    def test_area(self):
        # circle with center at origin
        geometry = Geometry(0., 0., 100., 0.0, 0., 0.2, False)

        # sector at 45 deg on circle
        vertex_x, vertex_y = geometry.initialize_sector_geometry(45./180.*np.pi)

        self.assertAlmostEqual(vertex_x[0], 65.21, 2)
        self.assertAlmostEqual(vertex_x[1], 79.70, 2)
        self.assertAlmostEqual(vertex_x[2], 62.03, 2)
        self.assertAlmostEqual(vertex_x[3], 75.81, 2)

        self.assertAlmostEqual(vertex_y[0], 62.03, 2)
        self.assertAlmostEqual(vertex_y[1], 75.81, 2)
        self.assertAlmostEqual(vertex_y[2], 65.21, 2)
        self.assertAlmostEqual(vertex_y[3], 79.70, 2)

        # sector at 0 deg on circle
        vertex_x, vertex_y = geometry.initialize_sector_geometry(0)

        self.assertAlmostEqual(vertex_x[0], 89.97, 2)
        self.assertAlmostEqual(vertex_x[1], 109.96, 2)
        self.assertAlmostEqual(vertex_x[2], 89.97, 2)
        self.assertAlmostEqual(vertex_x[3], 109.96, 2)

        self.assertAlmostEqual(vertex_y[0], -2.50, 2)
        self.assertAlmostEqual(vertex_y[1], -3.06, 2)
        self.assertAlmostEqual(vertex_y[2], 2.50, 2)
        self.assertAlmostEqual(vertex_y[3], 3.06, 2)

    def test_area2(self):
        # circle with center at 100.,100.
        geometry = Geometry(100., 100., 100., 0.0, 0., 0.2, False)

        # sector at 45 deg on circle
        vertex_x, vertex_y = geometry.initialize_sector_geometry(45./180.*np.pi)

        self.assertAlmostEqual(vertex_x[0], 165.21, 2)
        self.assertAlmostEqual(vertex_x[1], 179.70, 2)
        self.assertAlmostEqual(vertex_x[2], 162.03, 2)
        self.assertAlmostEqual(vertex_x[3], 175.81, 2)

        self.assertAlmostEqual(vertex_y[0], 162.03, 2)
        self.assertAlmostEqual(vertex_y[1], 175.81, 2)
        self.assertAlmostEqual(vertex_y[2], 165.21, 2)
        self.assertAlmostEqual(vertex_y[3], 179.70, 2)

        # sector at 225 deg on circle
        vertex_x, vertex_y = geometry.initialize_sector_geometry(225./180.*np.pi)

        self.assertAlmostEqual(vertex_x[0], 34.62, 2)
        self.assertAlmostEqual(vertex_x[1], 20.09, 2)
        self.assertAlmostEqual(vertex_x[2], 38.15, 2)
        self.assertAlmostEqual(vertex_x[3], 24.41, 2)

        self.assertAlmostEqual(vertex_y[0], 38.15, 2)
        self.assertAlmostEqual(vertex_y[1], 24.41, 2)
        self.assertAlmostEqual(vertex_y[2], 34.62, 2)
        self.assertAlmostEqual(vertex_y[3], 20.09, 2)

