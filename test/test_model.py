from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

from astropy.io import fits

from ellipse.ellipse import Ellipse
from ellipse.model import build_model

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