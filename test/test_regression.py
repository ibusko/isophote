from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest
import pyfits
from astropy.table import Table

from ellipse.ellipse import Ellipse
from ellipse.isophote import Isophote


class TestRegression(unittest.TestCase):
    '''
    Despite sharing attributes of a unit test, this code implements in fact
    regression testing of the Ellipse algorithm, against results obtained by
    the stsdas$analysis/isophote task 'ellipse'.

    The stsdas task was run on the M51.fits image and its results stored in
    file M51_table.fits. The code in here runs the Ellipse algorithm on the
    same M51.fits image, producing a list of Isophote instances. The contents
    of this list then get compared with the contents of M51_table.fits.

    For now, no comparisons are made. The data is just printed side by side
    to allow a visual comparison.
    '''
    def test_regression(self):

        table = Table.read('M51_table.fits')
        nrows = len(table['SMA'])
        print(table.columns)

        image = pyfits.open("M51.fits")
        test_data = image[0].data
        ellipse = Ellipse(test_data)
        isophote_list = ellipse.fit_image()

        # for key, value in t.meta.items():
        #     print('{0} = {1}'.format(key, value))

        format = "%5.2f  %6.1f %4.1f %6.1f %6.1f %5.3f  %5.1f %5.1f   %3d %2d"

        for row in range(nrows):
            iso = isophote_list[row]

            # data from Isophote
            sma_i = iso.sample.geometry.sma
            intens_i = iso.intens
            int_err_i = iso.int_err if iso.int_err else 0.
            pix_var_i = iso.pix_var if iso.pix_var else 0.
            rms_i = iso.rms if iso.rms else 0.
            ellip_i = iso.sample.geometry.eps if iso.sample.geometry.eps else 0.
            x0_i = iso.sample.geometry.x0
            y0_i = iso.sample.geometry.y0
            niter_i = iso.niter
            stop_i = iso.stop_code

            # ref data from table
            sma_t = table['SMA'][row]
            intens_t = table['INTENS'][row]
            int_err_t = table['INT_ERR'][row]
            pix_var_t = table['PIX_VAR'][row]
            rms_t = table['RMS'][row]
            ellip_t = table['ELLIP'][row]
            x0_t = table['X0'][row]
            y0_t = table['Y0'][row]
            niter_t = table['NITER'][row] if table['NITER'][row] else 0
            stop_t = table['STOP'][row] if table['STOP'][row] else -1

            print("* data "+format % (sma_i, intens_i, int_err_i, pix_var_i, rms_i, ellip_i, x0_i, y0_i, niter_i, stop_i))
            print("  ref  "+format % (sma_t, intens_t, int_err_t, pix_var_t, rms_t, ellip_t, x0_t, y0_t, niter_t, stop_t))
            print()


