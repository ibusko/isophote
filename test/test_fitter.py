from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import numpy as np
import pyfits

from util import build_test_data
from ellipse.harmonics import fit_1st_and_2nd_harmonics
from ellipse.sample import Sample
from ellipse.isophote import Isophote
from ellipse.fitter import Fitter


class TestFitter(unittest.TestCase):

    def test_gradient(self):

        test_data = build_test_data.build()

        sample = Sample(test_data, 40.)
        sample.update()

        self.assertAlmostEqual(sample.mean, 200.02, 2)
        self.assertAlmostEqual(sample.gradient, -4.222, 3)
        self.assertAlmostEqual(sample.gradient_error, 0.0003, 1)
        self.assertAlmostEqual(sample.gradient_relative_error, 7.45e-05, 2)
        self.assertAlmostEqual(sample.sector_area, 2.00, 2)

    def test_fitting_raw(self):
        # this test performs a raw (no Fitter), 1-step
        # correction in one single ellipse coefficient.

        test_data = build_test_data.build()

        # pick first guess ellipse that is off in just
        # one of the parameters (eps).
        sample = Sample(test_data, 40., eps=0.4)
        sample.update()
        s = sample.extract()

        y0, a1, b1, a2, b2 = fit_1st_and_2nd_harmonics(s[0], s[2])

        # when eps is off, b2 is the largest (in absolute value).
        self.assertGreater(abs(b2), abs(a1))
        self.assertGreater(abs(b2), abs(b1))
        self.assertGreater(abs(b2), abs(a2))

        correction = b2 * 2. * (1. - sample.geometry.eps) / sample.geometry.sma / sample.gradient
        new_eps = sample.geometry.eps - correction

        # got closer to test data (eps=0.2)
        self.assertAlmostEqual(new_eps, 0.21, 2)

    def test_fitting_small_radii(self):

        test_data = build_test_data.build()

        sample = Sample(test_data, 2.)
        fitter = Fitter(sample)

        isophote = fitter.fit()

        self.assertIsInstance(isophote, Isophote)
        self.assertEqual(isophote.ndata, 13)

    def test_fitting_eps(self):

        test_data = build_test_data.build()

        # initial guess is off in the eps parameter
        sample = Sample(test_data, 40., eps=0.4)
        fitter = Fitter(sample)

        isophote = fitter.fit()

        self.assertIsInstance(isophote, Isophote)
        g = isophote.sample.geometry
        self.assertGreaterEqual(g.eps, 0.19)
        self.assertLessEqual(g.eps, 0.21)

    def test_fitting_pa(self):

        test_data = build_test_data.build(pa=np.pi/4, noise=0.01)

        # initial guess is off in the pa parameter
        sample = Sample(test_data, 40)
        fitter = Fitter(sample)

        isophote = fitter.fit()

        g = isophote.sample.geometry
        self.assertGreaterEqual(g.pa, np.pi/4 - 0.05)
        self.assertLessEqual(g.pa, np.pi/4 + 0.05)

    def test_fitting_xy(self):

        test_data = build_test_data.build(x0=245, y0=245)

        # initial guess is off in the x0 and y0 parameters
        sample = Sample(test_data, 40)
        fitter = Fitter(sample)

        isophote = fitter.fit()

        g = isophote.sample.geometry
        self.assertGreaterEqual(g.x0, 245 - 1)
        self.assertLessEqual(g.x0,    245 + 1)
        self.assertGreaterEqual(g.y0, 245 - 1)
        self.assertLessEqual(g.y0,    245 + 1)

    def test_fitting_all(self):

        # build test image that is off from the defaults
        # assumed by the Sample constructor.
        POS = 250
        ANGLE = np.pi / 4
        EPS = 0.4
        test_data = build_test_data.build(x0=POS, y0=POS, eps=EPS, pa=ANGLE)

        # initial guess is off in all parameters. We find that the initial
        # guesses, especially for position angle, must be kinda close to the
        # actual value. 20% off max seems to work in this case of high SNR.
        sample = Sample(test_data, 40, position_angle=(1.2 * ANGLE))

        fitter = Fitter(sample)
        isophote = fitter.fit()

        self.assertEqual(isophote.stop_code, 0)

        g = isophote.sample.geometry
        self.assertGreaterEqual(g.x0, POS - 1.5)      # position within 1.5 pixel
        self.assertLessEqual(g.x0,    POS + 1.5)
        self.assertGreaterEqual(g.y0, POS - 1.5)
        self.assertLessEqual(g.y0,    POS + 1.5)
        self.assertGreaterEqual(g.eps, EPS - 0.01)    # eps within 0.01
        self.assertLessEqual(g.eps,    EPS + 0.01)
        self.assertGreaterEqual(g.pa, ANGLE - 0.05)   # pa within 5 deg
        self.assertLessEqual(g.pa,    ANGLE + 0.05)

    def test_m51(self):
        image = pyfits.open("data/M51.fits")
        test_data = image[0].data

        sample = Sample(test_data, 20., eps=0.1, position_angle=np.pi/4)
        fitter = Fitter(sample)
        isophote = fitter.fit()

        isophote.print()

        self.assertEqual(isophote.ndata, 113)
        self.assertAlmostEqual(isophote.intens, 732.5, 1)
