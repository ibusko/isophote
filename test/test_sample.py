from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import numpy as np

from util import build_test_data
from ellipse.integrator import MEAN, NEAREST_NEIGHBOR
from ellipse.sample import Sample


class TestSample(unittest.TestCase):

    def test_scatter(self):

        test_data = build_test_data.build(background=100., i0=00., noise=10.)

        astep = 0.1
        sma = 1.5

        sample = Sample(test_data, sma, astep=astep, integrmode=NEAREST_NEIGHBOR)
        sample.update()
        rms = np.std(sample.values[2])
        pix_var = rms * np.sqrt(sample.sector_area)
        print ('@@@@@@     line: 22  - ', rms, pix_var, sample.actual_points, sample.total_points, sample.sector_area)

        sample = Sample(test_data, sma, astep=astep)
        sample.update()
        rms = np.std(sample.values[2])
        pix_var = rms * np.sqrt(sample.sector_area)
        print ('@@@@@@     line: 28  - ', rms, pix_var, sample.actual_points, sample.total_points, sample.sector_area)

        sample = Sample(test_data, sma, astep=astep, integrmode=MEAN)
        sample.update()
        rms = np.std(sample.values[2])
        pix_var = rms * (np.sqrt(sample.sector_area))
        print ('@@@@@@     line: 34  - ', rms, pix_var, sample.actual_points, sample.total_points, sample.sector_area)

        # self.assertAlmostEqual(sample.mean, 200.02, 2)
        # self.assertAlmostEqual(sample.sector_area, 2.00, 2)


