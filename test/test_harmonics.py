from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from scipy.optimize import leastsq

import unittest

import pyfits

from ellipse.sample import Sample


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

        y0_1 = 100.
        a1_1 = 10.
        b1_1 = 5.
        a2_1 = 8.
        b2_1 = 2.

        optimize_func = lambda x: x[0] + x[1]*np.sin(E) + x[2]*np.cos(E) + x[3]*np.sin(2*E) + x[4]*np.cos(2*E) - data

        y0, a1, b1, a2, b2 = leastsq(optimize_func, [y0_1, a1_1, b1_1, a2_1, b2_1])[0]

        data_fit = y0 + a1*np.sin(E) + b1*np.cos(E) + a2*np.sin(2*E) + b2* np.cos(2*E) + 0.01 * np.random.randn(N)

        residual = data - data_fit

        self.assertAlmostEqual(np.mean(residual), 0.000, 2)
        self.assertAlmostEqual(np.std(residual),  0.015, 2)

