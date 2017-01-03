from __future__ import (absolute_import, division, print_function, unicode_literals)

import math
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

    Some quantities are compared in assert statements. These were designed to be
    executed only when the synth_highsnr.fits image is used as input. That way,
    we are mainly checking numerical differences that originate in the algorithms
    themselves, and not caused by noise. The quantities compared this way are:

      - mean intensity: 1% max difference for sma > 3 pixels, 5% otherwise
      - ellipticity: 1% max difference for sma > 3 pixels, 20% otherwise
      - position angle: 1 deg. max difference for sma > 3 pixels, 20 deg. otherwise

    For now, we can only check the bi-linear integration mode. The mean and median
    modes cannot be checked since the original 'ellipse' task has a bug that causes
    the creation of erroneous output tables. We need to write code that reads the
    standard output of 'ellipse' instead, captured from screen, and use it as
    reference.
    '''
    def test_regression(self):

        self._do_regression("M51")
        # self._do_regression("synth")
        # self._do_regression("synth_lowsnr")
        # self._do_regression("synth_highsnr")

    def _do_regression(self, name):

        table = Table.read(DATA + name + '_table.fits')
        # Original code in spp won't create the right table for the 'mean'.
        # integration mode. Use the screen output at synth_table_mean.txt to
        # compare results visually.
        #
        # table = Table.read(DATA + name + '_table_mean.fits')

        nrows = len(table['SMA'])
        print(table.columns)

        image = pyfits.open(DATA + name + ".fits")
        test_data = image[0].data
        ellipse = Ellipse(test_data)
        isophote_list = ellipse.fit_image()
        # isophote_list = ellipse.fit_image(integrmode=MEAN)

        format = "%5.2f  %6.1f    %8.3f %8.3f %8.3f        %7.5f  %6.2f   %06.2f %6.2f   %5.2f   %4d  %3d  %3d  %2d"

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
            rerr_i = iso.sample.gradient_relative_error if iso.sample.gradient_relative_error else 0.
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
            rerr_t = table['GRAD_R_ERR'][row]
            ndata_t = table['NDATA'][row]
            nflag_t = table['NFLAG'][row]
            niter_t = table['NITER'][row] if table['NITER'][row] else 0
            stop_t = table['STOP'][row] if table['STOP'][row] else -1

            # relative differences
            sma_d = (sma_i - sma_t) / sma_t * 100.
            intens_d = (intens_i - intens_t) / intens_t * 100.
            int_err_d = (int_err_i - int_err_t) / int_err_t * 100.
            pix_var_d = (pix_var_i - pix_var_t) / pix_var_t * 100.
            rms_d = (rms_i - rms_t) / rms_t * 100.
            ellip_d = (ellip_i - ellip_t) / ellip_t * 100.
            pa_d = pa_i - pa_t  # diff in angle is absolute
            x0_d = (x0_i - x0_t) / x0_t * 100.
            y0_d = (y0_i - y0_t) / y0_t * 100.
            rerr_d = rerr_i - rerr_t  # diff in relative error is absolute
            ndata_d = (ndata_i - ndata_t) / ndata_t * 100.
            nflag_d = 0
            niter_d = 0
            stop_d = 0 if stop_i == stop_t else -1

            print("* data "+format % (sma_i, intens_i, int_err_i, pix_var_i, rms_i, ellip_i, pa_i, x0_i, y0_i, rerr_i, ndata_i, nflag_i, niter_i, stop_i))
            print("  ref  "+format % (sma_t, intens_t, int_err_t, pix_var_t, rms_t, ellip_t, pa_t, x0_t, y0_t, rerr_t, ndata_t, nflag_t, niter_t, stop_t))
            print("  diff "+format % (sma_d, intens_d, int_err_d, pix_var_d, rms_d, ellip_d, pa_d, x0_d, y0_d, rerr_d, ndata_d, nflag_d, niter_d, stop_d))
            print()

            if name == "synth_highsnr":
                if sma_i > 3.:
                    self.assertLessEqual(abs(intens_d), 1.)
                else:
                    self.assertLessEqual(abs(intens_d), 5.)

                if not math.isnan(ellip_d):
                    if sma_i > 3.:
                        self.assertLessEqual(abs(ellip_d), 1.)   #  1%
                    else:
                        self.assertLessEqual(abs(ellip_d), 20.)  #  20%
                if not math.isnan(pa_d):
                    if sma_i > 3.:
                        self.assertLessEqual(abs(pa_d), 1.)      #  1 deg.
                    else:
                        self.assertLessEqual(abs(pa_d), 20.)     #  20 deg.




