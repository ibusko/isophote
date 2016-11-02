from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import numpy as np
import pyfits

from ellipse.sample import Sample

class TestSampling(unittest.TestCase):

    def test_sampling(self):

        test_data = pyfits.open("M51.fits")
        test_data = test_data[0].data

        sample = Sample(test_data, 100.)
        s = sample.extract()


        self.assertEqual(len(s), 3)


        print ('@@@@@@     line: 19  - ', len(s))



    # def test_something(self):
    #
    #     self.assertEqual(True, False)
