from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

from astropy.table import Table


class TestTable(unittest.TestCase):

    def test_table(self):
        t = Table.read('M51_table.fits')

