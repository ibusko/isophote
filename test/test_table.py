from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

from astropy.table import Table


class TestTable(unittest.TestCase):

    def test_table(self):
        t = Table.read('M51_table.fits')

        for key, value in t.meta.items():
            print('{0} = {1}'.format(key, value))

        print(t.columns)

        row = 32   # sma = 0
        for column in t.columns:
            print("%-10s   %f" % (column, t[column][row]))



