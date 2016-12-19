from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import pyfits

from util import build_test_data
from ellipse.sample import Sample
from ellipse.fitter import Fitter


class TestIsophote(unittest.TestCase):

    def test_fit(self):

        # low noise image, fitted perfectly by sample.
        test_data = build_test_data.build(noise=1.E-10)
        sample = Sample(test_data, 40)
        fitter = Fitter(sample)
        iso = fitter.fit(maxit=400)

        self.assertTrue(iso.valid)
        self.assertTrue(iso.stop_code == 0 or iso.stop_code == 2)

        # fitted values
        self.assertLessEqual(iso.intens,    201., 2)
        self.assertGreaterEqual(iso.intens, 199., 2)
        self.assertLessEqual(iso.int_err,    0.150, 2)
        self.assertGreaterEqual(iso.int_err, 0.130, 2)
        self.assertLessEqual(iso.pix_var,    3.20, 2)
        self.assertGreaterEqual(iso.pix_var, 2.70, 2)
        self.assertLessEqual(iso.grad,      -4.00, 2)
        self.assertGreaterEqual(iso.grad,   -4.30, 2)

        # integrals
        self.assertLessEqual(iso.tflux_e,    1.85E6, 2)
        self.assertGreaterEqual(iso.tflux_e, 1.82E6, 2)
        self.assertLessEqual(iso.tflux_c,    2.025E6, 2)
        self.assertGreaterEqual(iso.tflux_c, 2.022E6, 2)

        # deviations from perfect ellipticity
        self.assertLessEqual(abs(iso.a3), 0.01, 2)
        self.assertLessEqual(abs(iso.b3), 0.01, 2)
        self.assertLessEqual(abs(iso.a4), 0.01, 2)
        self.assertLessEqual(abs(iso.b4), 0.01, 2)

    def test_m51(self):

        image = pyfits.open("data/M51.fits")
        test_data = image[0].data

        sample = Sample(test_data, 20)
        fitter = Fitter(sample)
        iso = fitter.fit()

        self.assertTrue(iso.valid)
        self.assertTrue(iso.stop_code == 0 or iso.stop_code == 2)

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
        self.assertAlmostEqual(iso.intens,  734.704, 3)
        self.assertAlmostEqual(iso.rms,      89.753, 2)
        self.assertAlmostEqual(iso.int_err,   8.481, 2)
        self.assertAlmostEqual(iso.pix_var, 126.93,  2)
        self.assertAlmostEqual(iso.grad,    -45.674, 3)

        # integrals
        self.assertLessEqual(iso.tflux_e,    1.12E6, 2)
        self.assertGreaterEqual(iso.tflux_e, 1.11E6, 2)
        self.assertLessEqual(iso.tflux_c,    1.28E6, 2)
        self.assertGreaterEqual(iso.tflux_c, 1.26E6, 2)

        # deviations from perfect ellipticity
        self.assertLessEqual(abs(iso.a3), 0.04, 2)
        self.assertLessEqual(abs(iso.b3), 0.03, 2)
        self.assertLessEqual(abs(iso.a4), 0.03, 2)
        self.assertLessEqual(abs(iso.b4), 0.02, 2)

    def test_m51_niter(self):
        # compares with old STSDAS task. In this task, the
        # default for the starting value of SMA is 10; it
        # fits with 20 iterations.
        image = pyfits.open("data/M51.fits")
        test_data = image[0].data

        sample = Sample(test_data, 10)
        fitter = Fitter(sample)
        iso = fitter.fit()

        self.assertTrue(iso.valid)
        self.assertEqual(iso.niter, 17)


