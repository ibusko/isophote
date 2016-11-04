from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import numpy as np
import pyfits

from ellipse.sample import Sample

class TestSampling(unittest.TestCase):

    def test_sampling(self):

        test_data = pyfits.open("test_image.fits")
        test_data = test_data[0].data

        sample = Sample(test_data, 40., eps=0.4)
        s = sample.extract()

        self.assertEqual(len(s), 3)
        self.assertEqual(len(s[0]), len(s[1]))
        self.assertEqual(len(s[0]), len(s[2]))

        # values for image test_image.fits, sma=40., eps=0.4
        self.assertEqual(len(s[0]), 126)
        # intensities
        self.assertAlmostEqual(np.mean(s[2]), 0.1717,  3)
        self.assertAlmostEqual(np.std(s[2]),  0.00097, 5)
        # radii
        self.assertAlmostEqual(np.max(s[1]), 40.0, 2)
        self.assertAlmostEqual(np.min(s[1]), 24.0, 2)

