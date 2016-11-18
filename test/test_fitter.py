from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import numpy as np

from util import build_test_data
from ellipse.harmonics import fit_harmonics
from ellipse.sample import Sample
from ellipse.fitter import Fitter


class TestEllipse(unittest.TestCase):

    def test_gradient(self):

        test_data = build_test_data.build()

        sample = Sample(test_data, 40.)
        sample.update()

        self.assertAlmostEqual(sample.mean, 200.166,  3)
        self.assertAlmostEqual(sample.gradient, -4.178,  3)

    def test_fitting_raw(self):
        # this test performs a raw (no Fitter), 1-step
        # correction in one single ellipse coefficient.

        test_data = build_test_data.build()

        # pick first guess ellipse that is off in just
        # one of the parameters (eps).
        sample = Sample(test_data, 40., eps=0.3)
        sample.update()
        s = sample.extract()

        y0, a1, b1, a2, b2 = fit_harmonics(s[0], s[2])

        # when eps is off, b2 is the largest (in absolute value).
        correction = b2 * 2. * (1. - sample.geometry.eps) / sample.geometry.sma / sample.gradient
        new_eps = sample.geometry.eps - correction

        # got closer to test data (eps=0.2)
        self.assertAlmostEqual(new_eps, 0.19, 2)

    def test_fitting_eps(self):

        test_data = build_test_data.build()

        # initial guess is off in the eps parameter
        sample = Sample(test_data, 40., eps=0.4)
        fitter = Fitter(sample)

        sample = fitter.fit()

        self.assertIsInstance(sample, Sample)
        self.assertGreaterEqual(sample.geometry.eps, 0.19)
        self.assertLessEqual(sample.geometry.eps, 0.21)

    def test_fitting_pa(self):

        test_data = build_test_data.build(pa=np.pi/4, noise=0.01)

        # initial guess is off in the pa parameter
        sample = Sample(test_data, 40)
        fitter = Fitter(sample)

        sample = fitter.fit()

        self.assertGreaterEqual(sample.geometry.pa, np.pi/4 - 0.05)
        self.assertLessEqual(sample.geometry.pa, np.pi/4 + 0.05)

    def test_fitting_xy(self):

        test_data = build_test_data.build(x0=245, y0=245)

        # initial guess is off in the x0 and y0 parameters
        sample = Sample(test_data, 40)
        fitter = Fitter(sample)

        sample = fitter.fit()

        self.assertGreaterEqual(sample.geometry.x0, 245 - 1)
        self.assertLessEqual(sample.geometry.x0, 245 + 1)
        self.assertGreaterEqual(sample.geometry.y0, 245 - 1)
        self.assertLessEqual(sample.geometry.y0, 245 + 1)

    def test_fitting_all(self):

        test_data = build_test_data.build(x0=245, y0=245, eps=0.4, pa=np.pi/4)

        # initial guess is off in all parameters
        sample = Sample(test_data, 40)
        fitter = Fitter(sample)

        sample = fitter.fit()

        self.assertGreaterEqual(sample.geometry.x0, 245 - 1)
        self.assertLessEqual(sample.geometry.x0, 245 + 1)
        self.assertGreaterEqual(sample.geometry.y0, 245 - 1)
        self.assertLessEqual(sample.geometry.y0, 245 + 1)
        self.assertGreaterEqual(sample.geometry.eps, 0.39)
        self.assertLessEqual(sample.geometry.eps, 0.41)
        self.assertGreaterEqual(sample.geometry.pa, np.pi/4 - 0.05)
        self.assertLessEqual(sample.geometry.pa, np.pi/4 + 0.05)
