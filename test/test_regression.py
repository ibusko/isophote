from __future__ import (absolute_import, division, print_function, unicode_literals)

import unittest

import pyfits
import numpy as np

from astropy.table import Table

from ellipse.ellipse import Ellipse
from ellipse.integrator import BI_LINEAR, MEAN

DATA = "data/"


class TestRegression(unittest.TestCase):
    '''
    Despite being cast as a unit test, this code implements in fact
    regression testing of the Ellipse algorithm, against results obtained by
    the stsdas$analysis/isophote task 'ellipse'.

    The stsdas task was run on test images and results were stored in tables.
    The code in here runs the Ellipse algorithm on the same images, producing
    a list of Isophote instances. The contents of this list then get compared
    with the contents of the corresponding table.

    For now, no comparisons are calculated and displayed. The data is just
    printed side by side at stdout to allow a visual comparison.
    '''
    def test_regression(self):

        # self._do_regression("M51")
        self._do_regression("synth")
        # self._do_regression("synth_lowsnr")

    def _do_regression(self, name):

        table = Table.read(DATA + name + '_table.fits')
        # table = Table.read(DATA + name + '_table_mean.fits')
        nrows = len(table['SMA'])
        print(table.columns)

        image = pyfits.open(DATA + name + ".fits")
        test_data = image[0].data
        ellipse = Ellipse(test_data)
        isophote_list = ellipse.fit_image()
        # isophote_list = ellipse.fit_image(integrmode=MEAN)

        # for key, value in t.meta.items():
        #     print('{0} = {1}'.format(key, value))

        format = "%5.2f  %6.1f    %8.3f %8.3f %8.3f        %5.3f  %6.2f   %06.2f %6.2f      %4d  %3d  %3d  %2d"

        for row in range(nrows):
            try:
                iso = isophote_list[row]
            except IndexError:
                break

            # data from Isophote
            sma_i = iso.sample.geometry.sma
            intens_i = iso.intens
            int_err_i = iso.int_err if iso.int_err else 0.
            pix_var_i = iso.pix_var if iso.pix_var else 0.
            rms_i = iso.rms if iso.rms else 0.
            ellip_i = iso.sample.geometry.eps if iso.sample.geometry.eps else 0.
            pa_i = iso.sample.geometry.pa if iso.sample.geometry.pa else 0.
            x0_i = iso.sample.geometry.x0
            y0_i = iso.sample.geometry.y0
            ndata_i = iso.ndata
            nflag_i = iso.nflag
            niter_i = iso.niter
            stop_i = iso.stop_code

            # convert to old code reference system
            pa_i = (pa_i - np.pi/2) / np.pi * 180.
            x0_i += 1
            y0_i += 1

            # ref data from table
            sma_t = table['SMA'][row]
            intens_t = table['INTENS'][row]
            int_err_t = table['INT_ERR'][row]
            pix_var_t = table['PIX_VAR'][row]
            rms_t = table['RMS'][row]
            ellip_t = table['ELLIP'][row]
            pa_t = table['PA'][row]
            x0_t = table['X0'][row]
            y0_t = table['Y0'][row]
            ndata_t = table['NDATA'][row]
            nflag_t = table['NFLAG'][row]
            niter_t = table['NITER'][row] if table['NITER'][row] else 0
            stop_t = table['STOP'][row] if table['STOP'][row] else -1

            print("* data "+format % (sma_i, intens_i, int_err_i, pix_var_i, rms_i, ellip_i, pa_i, x0_i, y0_i, ndata_i, nflag_i, niter_i, stop_i))
            print("  ref  "+format % (sma_t, intens_t, int_err_t, pix_var_t, rms_t, ellip_t, pa_t, x0_t, y0_t, ndata_t, nflag_t, niter_t, stop_t))
            print()


