from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

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


