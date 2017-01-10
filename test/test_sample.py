from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

from util import build_test_data
from ellipse.integrator import MEDIAN, MEAN, BI_LINEAR, NEAREST_NEIGHBOR
from ellipse.sample import Sample
from ellipse.isophote import Isophote


class TestSample(unittest.TestCase):

    def test_scatter(self):
        '''
        Checks that the pixel standard deviation can be reliably estimated
        from the rms scatter and the sector area.

        The test data is just a flat image with noise. No galaxy. We define 
        the noise rms and then compare how close the pixel std dev estimated 
        at extraction matches this input noise.
        '''

        self.test_data = build_test_data.build(background=100., i0=0., noise=10.)

        self._doit(NEAREST_NEIGHBOR, 8., 11.)
        self._doit(BI_LINEAR,        8., 11.)
        self._doit(MEAN,             8., 11.)
        self._doit(MEDIAN,           7., 13.) # the median is not so good at estimating rms

    def _doit(self, integrmode, amin, amax):
        sample = Sample(self.test_data, 50., astep=0.2, integrmode=integrmode)
        sample.update()
        iso = Isophote(sample, 0, True, 0)

        self.assertLess(iso.pix_stddev, amax)
        self.assertGreater(iso.pix_stddev, amin)



