from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

from ellipse.harmonics import fit_upper_harmonic


def print_header(verbose=False):
    if verbose:
        print('#')
        print('# Semi-    Isophote      Ellipticity   Position    Grad.  Data  Flag  Iter. Stop')
        print('# major      mean                       Angle       rel.                    code')
        print('# axis     intensity                               error')
        print('#(pixel)                               (degree)')
        print('#')


class Isophote:

    def __init__(self, sample, niter, valid, stop_code):
        '''
        Container that helps in segregating information directly related to
        the sample (sampled intensities along an elliptical path on the image),
        from isophote-specific information.

        Parameters:
        ----------
        :param sample: instance of Sample
            the sample information
        :param niter: int
            number of iterations used to fit the isophote
        :param valid: boolean
            status of the fitting operation
        :param stop_code: int
            stop code:
                0 - normal.
                1 - less than pre-specified fraction of the extracted data
                    points are valid.
                2 - exceeded maximum number of iterations.
                3 - singular matrix in harmonic fit, results may not be valid.
                    Also signals insufficient number of data points to fit.
                4 - small or wrong gradient, or ellipse diverged. Subsequent
                    ellipses at larger or smaller semi-major axis may have the
                    same constant geometric parameters. It's also used when the
                    user turns off the fitting algorithm via the 'maxrit'
                    fitting parameter (see Ellipse class).
                5 - ellipse diverged; not even the minimum number of iterations
                    could be executed. Subsequent ellipses at larger or smaller
                    semi-major axis may have the same constant geometric
                    parameters.
               -1 - internal use.

        Attributes:
        -----------
        :param sma: float
            semi-major axis length (pixels)
        :param intens: float
            mean intensity value along the elliptical path
        :param rms: float
            root-mean-sq of intensity values along the elliptical path
        :param int_err: float
            error of the mean (rms / sqrt(# data points)))
        :param pix_stddev: float
            estimate of pixel standard deviation (rms * sqrt(average sector integration area))
        :param grad: float
            local radial intensity gradient
        :param grad_error: float
            measurement error of local radial intensity gradient
        :param grad_r_error: float
            relative error of local radial intensity gradient
        :param npix_e: int
            total number of valid pixels inside ellipse
        :param npix_c: int
            total number of valid pixels inside circle
        :param sarea: float
            average sector area on isophote (pixel)
        :param ndata: int
            number of valid data points on isophote
        :param nflag: int
            number of flagged data points on isophote
        '''
        self.sample = sample
        self.niter = niter
        self.valid = valid
        self.stop_code = stop_code

        self.intens = sample.mean
        self.rms = np.std(sample.values[2])
        self.int_err = self.rms / np.sqrt(sample.actual_points)
        self.pix_stddev = self.rms * np.sqrt(sample.sector_area)
        self.grad = sample.gradient
        self.grad_error = sample.gradient_error

        self.grad_r_error = sample.gradient_relative_error
        self.sarea = sample.sector_area
        self.ndata = sample.actual_points
        self.nflag = sample.total_points - sample.actual_points

        # flux contained inside ellipse and circle
        self.tflux_e, self.tflux_c, self.npix_e, self.npix_c = self._compute_fluxes()

        # deviations from perfect ellipticity
        try:
            c = fit_upper_harmonic(sample.values[0], sample.values[2], 3)
            self.a3 = c[1] / sample.geometry.sma /sample.gradient
            self.b3 = c[2] / sample.geometry.sma /sample.gradient
        except Exception as e: # we want to catch everything
            self.a3 = self.b3 = None
        try:
            c = fit_upper_harmonic(sample.values[0], sample.values[2], 4)
            self.a4 = c[1] / sample.geometry.sma /sample.gradient
            self.b4 = c[2] / sample.geometry.sma /sample.gradient
        except Exception as e: # we want to catch everything
            self.a4 = self.b4 = None

    def _compute_fluxes(self):
        # Compute integrated flux inside ellipse, as well as inside
        # circle defined by semi-major axis. Pixels in a square section
        # enclosing circle are scanned; the distance of each pixel to
        # the isophote center is compared both with the semi-major axis
        # length and with the length of the ellipse radius vector, and
        # integrals are updated if the pixel distance is smaller.

        # Compute limits of square array that encloses circle.
        sma = self.sample.geometry.sma
        x0 = self.sample.geometry.x0
        y0 = self.sample.geometry.y0
        xsize = self.sample.image.shape[1]
        ysize = self.sample.image.shape[0]

        imin = max(0, int(x0 - sma - 0.5) - 1)
        jmin = max(0, int(y0 - sma - 0.5) - 1)
        imax = min(xsize, int(x0 + sma + 0.5) + 1)
        jmax = min(ysize, int(y0 + sma + 0.5) + 1)

        # Integrate
        tflux_e = 0.
        tflux_c = 0.
        npix_e = 0
        npix_c = 0
        for j in range(jmin, jmax):
            for i in range(imin, imax):

                # radius of the circle and ellipse associated
                # with the given pixel.
                radius, angle = self.sample.geometry.to_polar(i, j)
                radius_e = self.sample.geometry.radius(angle)

                # pixel is inside circle with diameter given by sma
                if radius <= sma:
                    tflux_c += self.sample.image[j][i]
                    npix_c += 1

                # pixel is inside ellipse
                if radius <= radius_e:
                    tflux_e += self.sample.image[j][i]
                    npix_e += 1

        return tflux_e, tflux_c, npix_e, npix_c

    def __repr__(self):
        return "sma=%7.2f" % (self.sma)

    def print(self, verbose=False):
        if verbose:
            if self.grad_r_error:
                s = "%7.2f   %9.2f       % 5.3f       %6.2f      %5.3f  %4i %4i  %4i  %4i"% (self.sample.geometry.sma,
                                                       self.intens,
                                                       self.sample.geometry.eps,
                                                       self.sample.geometry.pa / np.pi * 180.,
                                                       self.grad_r_error,
                                                       self.ndata,
                                                       self.nflag,
                                                       self.niter,
                                                       self.stop_code)
            else:
                s = "%7.2f   %9.2f       % 5.3f       %6.2f      None   %4i %4i  %4i  %4i"% (self.sample.geometry.sma,
                                                       self.intens,
                                                       self.sample.geometry.eps,
                                                       self.sample.geometry.pa / np.pi * 180.,
                                                       self.ndata,
                                                       self.nflag,
                                                       self.niter,
                                                       self.stop_code)
            print(s)

    def fix_geometry(self, isophote):
        '''
        This method should be called when the fitting goes berserk and delivers
        an isophote with bad geometry, such as with eps>1 or another meaningless
        situation. This is not a problem in itself when fitting any given isophote,
        but will create an error when the affected isophote is used as starting
        guess for the next fit.

        The method will set the geometry of the affected isophote to be identical
        with the geometry of the isophote used as input value.

        Parameters:
        ----------
        :param isophote: instance of Isophote
            the isophote where to take geometry information from
        '''
        self.sample.geometry.eps = isophote.sample.geometry.eps
        self.sample.geometry.pa  = isophote.sample.geometry.pa
        self.sample.geometry.x0  = isophote.sample.geometry.x0
        self.sample.geometry.y0  = isophote.sample.geometry.y0

    # These two methods are useful for sorting lists of instances. Note
    # that __lt__ is the python3 way of supporting sorting. This might
    # not work under python2.
    @property
    def sma(self):
        return self.sample.geometry.sma

    def __lt__(self, other):
        if hasattr(other, 'sma'):
            return self.sma < other.sma


class CentralPixel(Isophote):
    '''
    Container for the central pixel in the galaxy image.

    For convenience, the CentralPixel class inherits from
    the Isophote class, although it's not really a true
    isophote but just a single intensity value at the central
    position. Thus, most of its attributes are hardcoded to
    None, or other default value when appropriate.
    '''
    def __init__(self, sample):

        self.sample = sample
        self.niter = 0
        self.valid = True
        self.stop_code = 0

        self.intens = sample.mean

        self.rms = None
        self.int_err = None
        self.pix_stddev = None
        self.grad = None
        self.grad_error = None
        self.grad_r_error = None
        self.sarea = None
        self.ndata = sample.actual_points
        self.nflag = sample.total_points - sample.actual_points

        self.tflux_e = self.tflux_c = self.npix_e = self.npix_c = None

        self.a3 = self.b3 = None
        self.a4 = self.b4 = None

    def print(self, verbose=False):
        if verbose:
            s = "   0.00   %9.2f"% (self.intens)
            print(s)


class IsophoteList(Isophote):
    '''
    This class is a convenience container that provides the same attributes
    that the Isophote class offers, except that scalar attributes are replaced
    by numpy array attributes. These arrays reflect the values of the given
    attribute across the entire list of Isophote instances provided at constructor
    time.
    '''
    def __init__(self, iso_list):
        '''
        Builds an IsophoteList instance from a (python) list
        of Isophote instances.

        :param iso_list: list
            a list with Isophote instances
        '''
        self._list = iso_list

    def as_list(self):
        '''
        Returns the contents of the instance as a list of
        Isophote instances.

        :return: list
            list with Isophote instances.
        '''
        return self._list

    def _collect_as_array(self, attr_name):
        return np.array(self._collect_as_list(attr_name))

    def _collect_as_list(self, attr_name):
        result = []
        for k in range(len(self._list)):
            result.append(getattr(self._list[k], attr_name))
        return result

    @property
    def sample(self):
        return self._collect_as_list('sample')

    @property
    def sma(self):
        return self._collect_as_array('sma')
    @property
    def intens(self):
        return self._collect_as_array('intens')
    @property
    def rms(self):
        return self._collect_as_array('rms')
    @property
    def int_err(self):
        return self._collect_as_array('int_err')
    @property
    def pix_stddev(self):
        return self._collect_as_array('pix_stddev')
    @property
    def grad(self):
        return self._collect_as_array('grad')
    @property
    def grad_error(self):
        return self._collect_as_array('grad_error')
    @property
    def grad_r_error(self):
        return self._collect_as_array('grad_r_error')
    @property
    def sarea(self):
        return self._collect_as_array('sarea')
    @property
    def ndata(self):
        return self._collect_as_array('ndata')
    @property
    def nflag(self):
        return self._collect_as_array('nflag')
    @property
    def niter(self):
        return self._collect_as_array('niter')
    @property
    def valid(self):
        return self._collect_as_array('valid')
    @property
    def stop_code(self):
        return self._collect_as_array('stop_code')
    @property
    def tflux_e(self):
        return self._collect_as_array('tflux_e')
    @property
    def tflux_c(self):
        return self._collect_as_array('tflux_c')
    @property
    def npix_e(self):
        return self._collect_as_array('npix_e')
    @property
    def npix_c(self):
        return self._collect_as_array('npix_c')
    @property
    def a3(self):
        return self._collect_as_array('a3')
    @property
    def b3(self):
        return self._collect_as_array('b3')
    @property
    def a4(self):
        return self._collect_as_array('a4')
    @property
    def b4(self):
        return self._collect_as_array('b4')


