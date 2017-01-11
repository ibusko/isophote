from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

from astropy.io import fits

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
        self.assertLessEqual(iso.intens,        201., 2)
        self.assertGreaterEqual(iso.intens,     199., 2)
        self.assertLessEqual(iso.int_err,       0.0010, 2)
        self.assertGreaterEqual(iso.int_err,    0.0009, 2)
        self.assertLessEqual(iso.pix_stddev,    0.03, 2)
        self.assertGreaterEqual(iso.pix_stddev, 0.02, 2)
        self.assertLessEqual(abs(iso.grad),     4.25, 2)
        self.assertGreaterEqual(abs(iso.grad),  4.20, 2)

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

        image = fits.open("data/M51.fits")
        test_data = image[0].data

        sample = Sample(test_data, 21.44)
        fitter = Fitter(sample)
        iso = fitter.fit()

        self.assertTrue(iso.valid)
        self.assertTrue(iso.stop_code == 0 or iso.stop_code == 2)

        # geometry
        g = iso.sample.geometry
        self.assertGreaterEqual(g.x0,  257 - 1.5)   # position within 1.5 pixel
        self.assertLessEqual(g.x0,     257 + 1.5)
        self.assertGreaterEqual(g.y0,  259 - 1.5)
        self.assertLessEqual(g.y0,     259 + 2.0)
        self.assertGreaterEqual(g.eps, 0.19 - 0.05) # eps within 0.05
        self.assertLessEqual(g.eps,    0.19 + 0.05)
        self.assertGreaterEqual(g.pa,  0.62 - 0.05) # pa within 5 deg
        self.assertLessEqual(g.pa,     0.62 + 0.05)

        # fitted values
        self.assertAlmostEqual(iso.intens,     674.1,  1)
        self.assertAlmostEqual(iso.rms,         85.38, 2)
        self.assertAlmostEqual(iso.int_err,      7.79, 2)
        self.assertAlmostEqual(iso.pix_stddev, 120.7,  1)
        self.assertAlmostEqual(iso.grad,       -34.49, 2)

        # integrals
        self.assertLessEqual(iso.tflux_e,    1.21E6, 2)
        self.assertGreaterEqual(iso.tflux_e, 1.20E6, 2)
        self.assertLessEqual(iso.tflux_c,    1.38E6, 2)
        self.assertGreaterEqual(iso.tflux_c, 1.36E6, 2)

        # deviations from perfect ellipticity
        self.assertLessEqual(abs(iso.a3), 0.05, 2)
        self.assertLessEqual(abs(iso.b3), 0.05, 2)
        self.assertLessEqual(abs(iso.a4), 0.05, 2)
        self.assertLessEqual(abs(iso.b4), 0.05, 2)

    def test_m51_niter(self):
        # compares with old STSDAS task. In this task, the
        # default for the starting value of SMA is 10; it
        # fits with 20 iterations.
        image = fits.open("data/M51.fits")
        test_data = image[0].data

        sample = Sample(test_data, 10)
        fitter = Fitter(sample)
        iso = fitter.fit()

        self.assertTrue(iso.valid)
        self.assertEqual(iso.niter, 20)


