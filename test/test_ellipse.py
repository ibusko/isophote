from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import numpy as np

from util import build_test_data
from ellipse.geometry import Geometry
from ellipse.fitter import TOO_MANY_FLAGGED
from ellipse.ellipse import Ellipse, FIXED_ELLIPSE, FAILED_FIT
from ellipse.isophote import Isophote

from util.build_test_data import DEFAULT_POS, DEFAULT_SIZE, DEFAULT_EPS

# define an off-center position and a tilted sma
POS = DEFAULT_POS + DEFAULT_SIZE / 4
PA = 10. / 180. * np.pi

# build off-center test data. Make the first guess
# geometry slightly offset from the actual image.
OFFSET_GALAXY = build_test_data.build(x0=POS, y0=POS, pa=PA, noise=1.E-12)
OFFSET_GEOMETRY = Geometry(POS+5, POS+5, 10., DEFAULT_EPS, PA, 0.1, False)


class TestEllipse(unittest.TestCase):

    def test_basic(self):
        # centered, tilted galaxy.
        test_data = build_test_data.build(pa=PA)

        ellipse = Ellipse(test_data)
        isophote_list = ellipse.fit_image()

        self.assertIsInstance(isophote_list, list)
        self.assertGreater(len(isophote_list), 1)
        self.assertIsInstance(isophote_list[0], Isophote)

        # verify that the list is properly sorted in sem-major axis length
        self.assertGreater(isophote_list[-1], isophote_list[0])

        # the fit should stop where gradient looses reliability.
        self.assertEqual(len(isophote_list), 67)
        self.assertEqual(isophote_list[-1].stop_code, FAILED_FIT)

    def test_offcenter_fail(self):
        # A first guess ellipse that is centered in the image frame.
        # This should result in failure since the real galaxy
        # image is off-center by a large offset.
        ellipse = Ellipse(OFFSET_GALAXY)
        isophote_list = ellipse.fit_image()

        self.assertEqual(len(isophote_list), 0)

    def test_offcenter_fit(self):
        # A first guess ellipse that is roughly centered on the
        # offset galaxy image.
        ellipse = Ellipse(OFFSET_GALAXY, geometry=OFFSET_GEOMETRY)
        isophote_list = ellipse.fit_image()

        # the fit should stop when too many potential sample
        # points fall outside the image frame.
        self.assertEqual(len(isophote_list), 63)
        self.assertEqual(isophote_list[-1].stop_code, TOO_MANY_FLAGGED)

    def test_offcenter_go_beyond_frame(self):
        # Same as before, but now force the fit to goo
        # beyond the image frame limits.
        ellipse = Ellipse(OFFSET_GALAXY, geometry=OFFSET_GEOMETRY)
        isophote_list = ellipse.fit_image(maxsma=400.)

        # the fit should go to maxsma, but with fixed geometry
        self.assertEqual(len(isophote_list), 71)
        self.assertEqual(isophote_list[-1].stop_code, FIXED_ELLIPSE)
