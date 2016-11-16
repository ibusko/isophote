from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from scipy.optimize import leastsq

import unittest

from util import build_test_data

from ellipse.sample import Sample
from ellipse.harmonics import fit_harmonics, harmonic_function


class TestHarmonics(unittest.TestCase):

    def test_harmonics_1(self):

        # this is an almost as-is example taken from stackoverflow

        N = 100 # number of data points
        t = np.linspace(0, 4*np.pi, N)

        # create artificial data with noise:
        # mean = 0.5, amplitude = 3., phase = 0.1, noise-std = 0.01
        data = 3.0 * np.sin(t + 0.1) + 0.5 + 0.01 * np.random.randn(N)

        # first guesses for harmonic parameters
        guess_mean = np.mean(data)
        guess_std = 3 * np.std(data)/(2**0.5)
        guess_phase = 0

        # Minimize the difference between the actual data and our "guessed" parameters
        optimize_func = lambda x: x[0] * np.sin(t+x[1]) + x[2] - data

        est_std, est_phase, est_mean = leastsq(optimize_func, [guess_std, guess_phase, guess_mean])[0]

        # recreate the fitted curve using the optimized parameters
        data_fit = est_std * np.sin(t+est_phase) + est_mean
        residual = data - data_fit

        self.assertAlmostEqual(np.mean(residual), 0.000000, 6)
        self.assertAlmostEqual(np.std(residual),  0.01, 2)

    def test_harmonics_2(self):

        # this uses the actual functional form used for fitting ellipses

        N = 100
        E = np.linspace(0, 4*np.pi, N)

        y0_0 = 100.
        a1_0 = 10.
        b1_0 = 5.
        a2_0 = 8.
        b2_0 = 2.
        data = y0_0 + a1_0 * np.sin(E) + b1_0 * np.cos(E) + a2_0 * np.sin(2*E) + b2_0 * np.cos(2*E) + 0.01 * np.random.randn(N)

        y0, a1, b1, a2, b2 = fit_harmonics(E, data)
        data_fit = y0 + a1*np.sin(E) + b1*np.cos(E) + a2*np.sin(2*E) + b2* np.cos(2*E) + 0.01 * np.random.randn(N)
        residual = data - data_fit

        self.assertAlmostEqual(np.mean(residual), 0.000, 2)
        self.assertAlmostEqual(np.std(residual),  0.015, 2)

    def test_fit_sample_1(self):

        # major axis parallel to X image axis
        test_data = build_test_data.build()

        sample = Sample(test_data, 40.)
        s = sample.extract()

        y0, a1, b1, a2, b2 = fit_harmonics(s[0], s[2])

        self.assertAlmostEqual(y0, 200.153, 3)
        self.assertAlmostEqual(a1, 0.03632, 4)
        self.assertAlmostEqual(b1, 0.07501, 4)
        self.assertAlmostEqual(a2, 0.09122, 4)
        self.assertAlmostEqual(b2, 0.2617, 4)

        # check that harmonics subtract nicely
        model = harmonic_function(s[0], y0, a1, b1, a2, b2)
        residual = s[2] - model

        self.assertTrue(np.mean(residual) < 1.E-4)
        self.assertAlmostEqual(np.std(residual),  2.06, 2)

    def test_fit_sample_2(self):

        # major axis tilted 45 deg wrt X image axis
        test_data = build_test_data.build(pa=np.pi/4)

        sample = Sample(test_data, 40., eps=0.4)
        s = sample.extract()

        y0, a1, b1, a2, b2 = fit_harmonics(s[0], s[2])

        self.assertAlmostEqual(y0, 246.354, 3)
        self.assertAlmostEqual(a1, -1.08690, 4)
        self.assertAlmostEqual(b1, -0.1685, 3)
        # these drive the correction for position angle
        self.assertAlmostEqual(a2, 28.4540, 4)
        self.assertAlmostEqual(b2, -65.2513, 4)

    def test_fit_sample_3(self):

        test_data = build_test_data.build()

        # initial guess is rounder than actual image
        sample = Sample(test_data, 40., eps=0.1)
        s = sample.extract()

        y0, a1, b1, a2, b2 = fit_harmonics(s[0], s[2])

        self.assertAlmostEqual(y0, 189.273, 3)
        self.assertAlmostEqual(a1, 0.1081, 4)
        self.assertAlmostEqual(b1, -0.1407, 4)
        self.assertAlmostEqual(a2, 0.04396, 4)
        # this drives the correction for ellipticity
        self.assertAlmostEqual(b2, 9.8599, 4)

    def test_fit_sample_4(self):

        test_data = build_test_data.build()

        # initial guess for center is offset
        sample = Sample(test_data, x0=220., y0=210., sma=40.)
        s = sample.extract()

        y0, a1, b1, a2, b2 = fit_harmonics(s[0], s[2])

        self.assertAlmostEqual(y0, 152.869, 3)
        self.assertAlmostEqual(a1, 55.3241, 4)
        self.assertAlmostEqual(b1, 33.5510, 4)
        self.assertAlmostEqual(a2, 33.0182, 4)
        self.assertAlmostEqual(b2, -13.8363, 4)


