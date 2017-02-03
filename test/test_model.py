from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import numpy as np
from astropy.io import fits

from ellipse.geometry import Geometry
from ellipse.ellipse import Ellipse
from ellipse.model import build_model
from util.build_test_data import build

DATA = "data/"


class TestModel(unittest.TestCase):

    def test_model(self):
        name = "M51"
        test_data = fits.open(DATA + name + ".fits")
        image = test_data[0].data
        ellipse = Ellipse(image, verbose=True)
        isophote_list = ellipse.fit_image(verbose=True)

        model = build_model(image, isophote_list)

        self.assertEqual(image.shape, model.shape)

    def test_2(self):
        pixel_data = build(eps=0.5, pa=np.pi/3., noise=1.e-2)

        g = Geometry(256., 256., 10., 0.5, np.pi/3.)
        ellipse = Ellipse(pixel_data, geometry=g)
        isolist = ellipse.fit_image()
        model_image = build_model(pixel_data, isolist, fill=np.mean(pixel_data[0:50,0:50]))

