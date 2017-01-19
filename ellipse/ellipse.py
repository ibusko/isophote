from __future__ import (absolute_import, division, print_function, unicode_literals)

from ellipse.geometry import DEFAULT_STEP
from ellipse.integrator import BI_LINEAR
from ellipse.sample import Sample, CentralSample, DEFAULT_SCLIP
from ellipse.fitter import Fitter, CentralFitter, TOO_MANY_FLAGGED, \
    DEFAULT_CONVERGENCY, DEFAULT_MINIT, DEFAULT_MAXIT, DEFAULT_FFLAG, DEFAULT_MAXGERR
from ellipse.isophote import Isophote, IsophoteList, print_header


FIXED_ELLIPSE = 4
FAILED_FIT = 5


class Ellipse():
    '''
    This class provides the main access point to the isophote fitting algorithm.

    This algorithm is designed to fit elliptical isophotes to galaxy images. Its main input
    is a 2-dimensional numpy array with the image. The output is a list of instances of class
    Isophote. See the documentation for this class for details. For convenience, this list is
    packaged in an instance of class IsophoteList, which augments the list with isophote-specific
    features.

    During the fitting process, some of the isophote parameters are displayed in tabular form
    at the standard output. These parameters allow the user to monitor the fitting process.

    The image is measured using an iterative method described by Jedrzejewski (Mon.Not.R.Astr.Soc.,
    226, 747, 1987). Each isophote is fitted at a pre-defined, fixed semi-major axis length.
    The algorithm starts from a first guess elliptical isophote defined by approximate values for
    the X and Y center coordinates, ellipticity and position angle. Using these values, the
    image is sampled along an elliptical path, producing a 1-dimensional function that describes
    the dependency of intensity (pixel value) with angle (E). The function is stored as a set of
    1-D numpy arrays. The harmonic content of this function is analyzed by least-squares fitting
    to the function:

    y  =  y0 + A1 * sin(E) + B1 * cos(E) + A2 * sin(2 * E) + B2 * cos (2 * E)

    Each one of the harmonic amplitudes A1, B1, A2, B2 is related to a specific ellipse
    geometric parameter, in the sense that it conveys information regarding how much the current
    parameter value deviates from the "true" one. To compute this deviation, the image's local
    radial gradient has to be taken into account too. The algorithm picks up the largest amplitude
    among the four, estimates the local gradient and computes the corresponding increment in the
    associated ellipse parameter. That parameter is updated, and the image is resampled. This
    process is repeated until any one of the following criteria are met:

    1 - the largest harmonic amplitude is less than a given fraction of the rms residual of the
        intensity data around the harmonic fit.

    2 - a user-specified maximum number of iterations is reached.

    3 - more than a given fraction of the elliptical sample points have no valid data in then,
        either because they lie outside the image boundaries or because they where flagged out from
        the fit (see below).

    In any case, a minimum number of iterations is always performed. If iterations stop because
    of reasons 2 or 3 above, then those ellipse parameters that generated the lowest absolute
    values for harmonic amplitudes will be used. At this point, the image data sample coming from
    the best fit ellipse is fitted by the following function:

    y  =  y0  +  An * sin(n * E)  +  Bn * cos(n * E)

    with n = 3 and n = 4. The corresponding amplitudes (A3, B3, A4, B4), divided by the semi-major
    axis length and local intensity gradient, measure the isophote's deviations from perfect
    ellipticity (the amplitudes divided by semi-major axis and gradient, are the actual quantities
    stored in the output Isophote instance).

    The algorithm then measures the integrated intensity and the number of non-flagged pixels inside
    the elliptical isophote, and also inside the corresponding circle with same center and radius
    equal to the semi-major axis length. These parameters, some other associated parameters, and
    some auxiliary information, are stored in the Isophote instance.

    It must be emphasized that the algorithm was designed explicitly with a (elliptical) galaxy
    brightness distribution in mind. In particular, a well defined negative radial intensity
    gradient across the region being fitted is paramount for the achievement of stable solutions.
    Use of the algorithm in other types of images (e.g., planetary nebulae) may lead to inability
    to converge to any acceptable solution.

    After fitting the ellipse that corresponds to a given value of the semi-major axis (by the
    process described above), the axis length is incremented/decremented following a pre-defined
    rule. At each step, the starting, first guess ellipse parameters are taken from the previously
    fitted ellipse that has the closest semi-major axis length to the current one. On low surface
    brightness regions (i.e., those having large radii), the small values of the image radial
    gradient can induce large corrections and meaningless values for the ellipse parameters. The
    algorithm has capabilities to stop increasing semi-major axis based on several criteria, including
    signal-to-noise ratio.

    The 'ellipse' algorithm also provides a k-sigma clipping algorithm for cleaning deviant sample
    points at each isophote, thus improving convergency stability against any non-elliptical structure
    such as stars, spiral arms, HII regions, defects, etc.

    '''
    def __init__(self, image, geometry=None):
        '''
        Constructor

        :param image: np 2-D array
            image array
        :param geometry: instance of Geometry
            the optional geometry that describes the first ellipse to be fitted
        '''
        self.image = image
        self._geometry = geometry

    def fit_image(self, sma0 = 10.,
                          minsma     = 0.,
                          maxsma     = None,
                          step       = DEFAULT_STEP,
                          conver     = DEFAULT_CONVERGENCY,
                          minit      = DEFAULT_MINIT,
                          maxit      = DEFAULT_MAXIT,
                          fflag      = DEFAULT_FFLAG,
                          maxgerr    = DEFAULT_MAXGERR,
                          sclip      = DEFAULT_SCLIP,
                          nclip      = 0,
                          integrmode = BI_LINEAR,
                          linear     = False,
                          maxrit     = None,
                          verbose    = True):
        '''
        Main fitting method. Fits multiple isophotes on the image array passed
        to the constructor. This method basically loops over each one of the
        values of semi-major axis length (sma) constructed from the input parameters,
        and fits one isophote at each sma, returning the entire set of isophotes in
        a sorted IsophoteList instance.

        :param sma0: float, default = 10.
            starting value for the semi-major axis length (pixels). This can't be
            neither the minimum or the maximum, but something in between. The
            algorithm can't start from the very center of the galaxy image because
            the modelling of elliptical isophotes on that region is poor and it will
            diverge very easily if not tied to other previously fit isophotes. It can't
            start from the maximum value either because the maximum is not known
            beforehand, depending on signal-to-noise. The sma0 value should be selected
            such that the corresponding isophote has a good signal-to-noise ratio and
            a clear geometry.
        :param minsma:  float, default = 0.
            minimum value for the semi-major axis length (pixels).
        :param maxsma: float, default = None.
            maximum value for the semi-major axis length (pixels).
            When set to None, the algorithm will increase the semi
            major axis until one of several conditions will cause
            it to stop and revert to fit ellipses with sma < sma0.
        :param step: float, default = 0.1
            the step value being used to grow/shrink the semi-major
            axis length (pixels if 'linear=True', or relative value
            if 'linear=False')
        :param conver: float, default = 0.05
            main convergency criterion. Largest harmonic amplitude
            must be smaller than 'conver' times the fit rms.
        :param minit: int, default = 10
            minimum number of iterations to perform
        :param maxit: int, default = 50
            maximum number of iterations to perform
        :param fflag: float, default = 0.7
            acceptable fraction of flagged data points in sample.
            If the actual number of valid data points is smaller
            than this, stop iterating and return current Isophote.
            Flagged data points include points that lie outside
            the image frame, as well as points removed by sigma
            clipping.
        :param maxgerr: float, default = 0.5
            maximum acceptable relative error in the local radial
            intensity gradient.
        :param sclip: float, default = 3.0
            sigma-cliping criterion
        :param nclip: int, default = 0
            number of iterations in sigma-cliping algorithm.
            If zero, ignore sigma-clip.
        :param integrmode: string, default = 'bi-linear'
            integration mode, as defined in module integrator.py
        :param linear: boolean, default False
            semi-major axis growing/shrinking mode
        :param maxrit: float, default None
            maximum value of semi-major axis to perform an actual fit.
            Whenever the current semi-major axis length is larger than
            maxrit, the isophotes wil be just extracted using the current
            geometry, without being fitted. Ignored if None.
        :param verbose: boolean, default True
            print iteration info
        :return: IsophoteList instance
            this list stores fitted Isophote instances, sorted according
            to the semi-major axis length value.
        '''

        # multiple fitted isophotes will be stored here
        isophote_list = []

        print_header(verbose)

        # first, go from initial sma outwards until
        # hitting one of several stopping criteria.
        sma = sma0
        noiter = False
        while True:
            isophote = self.fit_isophote(sma, step, conver, minit, maxit, fflag, maxgerr,
                                         sclip, nclip, integrmode,
                                         linear, maxrit, noniterate=noiter,
                                         isophote_list=isophote_list)

            # check for failed fit.
            if isophote.stop_code < 0 or isophote.stop_code == TOO_MANY_FLAGGED:

                # in case the fit failed right at the outset, return an empty
                # list. This is the usual case when the user provides initial
                # guesses that are too way off to enable the fitting algorithm
                # to find any meaningful solution.
                if len(isophote_list) == 1:
                    return IsophoteList([])

                self._fix_last_isophote(isophote_list, -1)

                # get last isophote from the actual list, since the last
                # 'isophote' instance in this context may no longer be OK.
                isophote = isophote_list[-1]

                # if two consecutive isophotes failed to fit,
                # shut off iterative mode. Or, bail out and
                # change to go inwards.
                if len(isophote_list) > 1:
                    if (isophote.stop_code == FAILED_FIT and isophote_list[-2].stop_code == FAILED_FIT) \
                            or \
                        isophote.stop_code == TOO_MANY_FLAGGED:
                        if maxsma and maxsma > isophote.sma:
                            # if a maximum sma value was provided by user, and the
                            # current sma is smaller than maxsma, keep growing sma
                            # in non-iterative mode until reaching it.
                            noiter = True
                        else:
                            # if no maximum sma, stop growing and change
                            # to go inwards. Print from last kept isophote.
                            isophote.print(verbose)
                            break

            # reset variable from the actual list, since the last
            # 'isophote' instance may no longer be OK.
            isophote = isophote_list[-1]
            isophote.print(verbose)

            # update sma. If exceeded user-defined
            # maximum, bail out from this loop.
            sma = isophote.sample.geometry.update_sma(step)
            if maxsma and sma >= maxsma:
                break

        # reset sma so as to go inwards.
        first_isophote = isophote_list[0]
        sma, step = first_isophote.sample.geometry.reset_sma(step)

        # now, go from initial sma inwards towards center.
        while True:
            isophote = self.fit_isophote(sma, step, conver, minit, maxit, fflag, maxgerr,
                                         sclip, nclip,
                                         integrmode, linear, maxrit,
                                         going_inwards=True,
                                         isophote_list=isophote_list)

            # if abnormal condition, fix isophote but keep going.
            if isophote.stop_code < 0:
                self._fix_last_isophote(isophote_list, 0)

            # reset variable from the actual list, since the last
            # 'isophote' instance may no longer be OK.
            isophote = isophote_list[-1]
            isophote.print(verbose)

            # figure out next sma; if exceeded user-defined
            # minimum, or too small, bail out from this loop
            sma = isophote.sample.geometry.update_sma(step)
            if sma <= max(minsma, 0.5):
                break

        # if user asked for minsma=0, extract special isophote there
        if minsma == 0.0:
            isophote = self.fit_isophote(0.0, isophote_list=isophote_list)
            isophote.print(verbose)

        # sort list of isophotes according to sma
        isophote_list.sort()

        return IsophoteList(isophote_list)

    def fit_isophote(self, sma,
                            step          = DEFAULT_STEP,
                            conver        = DEFAULT_CONVERGENCY,
                            minit         = DEFAULT_MINIT,
                            maxit         = DEFAULT_MAXIT,
                            fflag         = DEFAULT_FFLAG,
                            maxgerr       = DEFAULT_MAXGERR,
                            sclip         = DEFAULT_SCLIP,
                            nclip         = 0,
                            integrmode    = BI_LINEAR,
                            linear        = False,
                            maxrit        = None,
                            noniterate    = False,
                            going_inwards = False,
                            isophote_list = None):
        '''
        Fit one isophote with a given semi-major axis length.

        The 'step' and 'linear' parameters are not used to actually
        grow or shrink the current fitting semi-major axis length.
        They are necessary nevertheless, so the sampling algorithm
        can know where to start the gradient computation, and also
        how to compute the elliptical sector areas (when area
        integration mode is selected).

        :param sma: float
            the semi-major axis length (pixels)
        :param step: float, default = 0.1
            the step value being used to grow/shrink the semi-major
            axis length (pixels)
        :param conver: float, default = 0.05
            main convergency criterion. Largest harmonic amplitude
            must be smaller than 'conver' times the fit rms.
        :param minit: int, default = 10
            minimum number of iterations to perform
        :param maxit: int, default = 50
            maximum number of iterations to perform
        :param fflag: float, default = 0.7
            acceptable fraction of flagged data points in sample.
            If the actual number of valid data points is smaller
            than this, stop iterating and return current Isophote.
            Flagged data points include points that lie outside
            the image frame, as well as points removed by sigma
            clipping.
        :param maxgerr: float, default = 0.5
            maximum acceptable relative error in the local radial
            intensity gradient.
        :param sclip: float, default = 3.0
            sigma-cliping criterion
        :param nclip: int, default = 0
            number of iterations in sigma-cliping algorithm.
            If zero, ignore sigma-clip.
        :param integrmode: string, default = 'bi-linear'
            integration mode, as defined in module integrator.py
        :param linear: boolean, default = False
            semi-major axis growing/shrinking mode
        :param maxrit: float, default None
            maximum value of semi-major axis to perform an actual fit.
            Whenever the current semi-major axis length is larger than
            maxrit, the isophote wil be just extracted using the current
            geometry, without fitting it. Ignored if None.
        :param noniterate: boolean, default False
            signals that the fitting algorithm should be bypassed and an
            isophote should be extracted with the geometry taken directly
            from the most recent Isophote instance stored in the
            'isophote_list' parameter.
            This parameter is mainly used when running the method in a loop
            over different values of semi-major axis length, and we want
            to change from iterative to non-iterative mode somewhere
            along the sequence of isophotes. When set to True, this
            parameter overrides the behavior associated with parameter
            'maxrit'.
        :param going_inwards: boolean, default False
            defines the sense of SMA growth. This is used by stopping
            criteria that depend on the gradient relative error.
        :param isophote_list: list, default = None
            fitted Isophote instance is appended to this list. Must
            be created and managed by the caller.
        :return: Isophote instance
            the fitted isophote. The fitted isophote is also appended
            to the input list passed via parameter 'isophote_list'.
        '''
        # if available, geometry from last fitted isophote will be
        # used as initial guess for next isophote.
        geometry = self._geometry
        if isophote_list is not None and len(isophote_list) > 0:
            geometry = isophote_list[-1].sample.geometry

        # do the fit.
        if noniterate or (maxrit and sma > maxrit):
            isophote = self._non_iterative(sma, step, linear, geometry,
                                           sclip, nclip)
        else:
            isophote = self._iterative(sma, step, linear, geometry, sclip, nclip,
                                       integrmode, conver, minit, maxit, fflag,
                                       maxgerr, going_inwards)

        # store result in list
        if isophote_list is not None and isophote.valid:
            isophote_list.append(isophote)

        return isophote

    def _iterative(self, sma, step, linear, geometry, sclip, nclip, integrmode,
                   conver, minit, maxit, fflag, maxgerr, going_inwards=False):
        if sma > 0.:
            # iterative fitter
            sample = Sample(self.image, sma,
                            astep=step,
                            sclip=sclip,
                            nclip=nclip,
                            linear_growth=linear,
                            geometry=geometry,
                            integrmode=integrmode)
            fitter = Fitter(sample)
        else:
            # sma == 0 requires special handling.
            sample = CentralSample(self.image, 0.0, geometry=geometry)
            fitter = CentralFitter(sample)

        isophote = fitter.fit(conver, minit, maxit, fflag, maxgerr, going_inwards)

        return isophote

    def _non_iterative(self, sma, step, linear, geometry, sclip, nclip):
        sample = Sample(self.image, sma,
                        astep=step,
                        sclip=sclip,
                        nclip=nclip,
                        linear_growth=linear,
                        geometry=geometry)
        sample.update()

        # build isophote without iterating with a Fitter
        isophote = Isophote(sample, 0, True, FIXED_ELLIPSE)

        return isophote

    def _fix_last_isophote(self, isophote_list, index):
        if len(isophote_list) > 0:
            isophote = isophote_list.pop()

            # check if isophote is bad; if so, fix its geometry
            # to be like the geometry of the index-th isophote
            # in list.
            isophote.fix_geometry(isophote_list[index])

            # force new extraction of raw data, since
            # geometry changed.
            isophote.sample.values = None
            isophote.sample.update()

            # we take the opportunity to change an eventual
            # negative stop code to its' positive equivalent.
            code = FAILED_FIT if isophote.stop_code < 0 else isophote.stop_code

            # build new instance so it can have its attributes
            # populated from the updated sample attributes.
            new_isophote = Isophote(isophote.sample, isophote.niter, isophote.valid, code)

            # add new isophote to list
            isophote_list.append(new_isophote)




