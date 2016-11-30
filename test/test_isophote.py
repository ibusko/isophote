from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import pyfits

from util import build_test_data
from ellipse.sample import Sample
from ellipse.fitter import Fitter


class TestIsophote(unittest.TestCase):

    def test_fit(self):

        # low noise image, fitted perfectly by sample.
        test_data = build_test_data.build()
        sample = Sample(test_data, 40)
        fitter = Fitter(sample)
        iso = fitter.fit()

        # fitted values
        self.assertAlmostEqual(iso.intens,  200.1659, 4)
        self.assertAlmostEqual(iso.rms,     2.073, 3)
        self.assertAlmostEqual(iso.int_err, 0.139, 3)
        self.assertAlmostEqual(iso.pix_var, 2.932, 3)
        self.assertAlmostEqual(iso.grad,   -4.178, 3)

        # integrals
        self.assertLessEqual(iso.tflux_e,    1.85E6, 2)
        self.assertGreaterEqual(iso.tflux_e, 1.82E6, 2)
        self.assertLessEqual(iso.tflux_c,    2.025E6, 2)
        self.assertGreaterEqual(iso.tflux_c, 2.022E6, 2)

    def test_m51(self):

        image = pyfits.open("M51.fits")
        test_data = image[0].data

        sample = Sample(test_data, 20)
        fitter = Fitter(sample)
        iso = fitter.fit()

        # geometry
        g = iso.sample.geometry
        self.assertGreaterEqual(g.x0,  257 - 1.5)   # position within 1.5 pixel
        self.assertLessEqual(g.x0,     257 + 1.5)
        self.assertGreaterEqual(g.y0,  259 - 1.5)
        self.assertLessEqual(g.y0,     259 + 1.5)
        self.assertGreaterEqual(g.eps, 0.19 - 0.01) # eps within 0.01
        self.assertLessEqual(g.eps,    0.19 + 0.01)
        self.assertGreaterEqual(g.pa,  0.62 - 0.05) # pa within 5 deg
        self.assertLessEqual(g.pa,     0.62 + 0.05)

        # fitted values
        self.assertAlmostEqual(iso.intens,  740.150, 3)
        self.assertAlmostEqual(iso.rms,     160.61, 2)
        self.assertAlmostEqual(iso.int_err, 15.18, 2)
        self.assertAlmostEqual(iso.pix_var, 227.14, 2)
        self.assertAlmostEqual(iso.grad,   -36.133, 3)

        # integrals
        self.assertLessEqual(iso.tflux_e,    1.12E6, 2)
        self.assertGreaterEqual(iso.tflux_e, 1.11E6, 2)
        self.assertLessEqual(iso.tflux_c,    1.28E6, 2)
        self.assertGreaterEqual(iso.tflux_c, 1.26E6, 2)


