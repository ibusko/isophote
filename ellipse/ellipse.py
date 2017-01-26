from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
from numpy import ma as ma

from ellipse.geometry import Geometry, DEFAULT_STEP, DEFAULT_EPS
from ellipse.integrator import BI_LINEAR
from ellipse.sample import Sample, CentralSample, DEFAULT_SCLIP
from ellipse.fitter import Fitter, CentralFitter, TOO_MANY_FLAGGED, \
    DEFAULT_CONVERGENCY, DEFAULT_MINIT, DEFAULT_MAXIT, DEFAULT_FFLAG, DEFAULT_MAXGERR
from ellipse.isophote import Isophote, IsophoteList, print_header


FIXED_ELLIPSE = 4
FAILED_FIT = 5
DEFAULT_THRESHOLD = 1.0


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
        the fit by sigma-clipping.

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
    equal to the semi-major axis length. These parameters, their errors, other associated parameters,
    and auxiliary information, are stored in the Isophote instance.

    Errors in intensity and local gradient are obtained directly from the rms scatter of intensity
    data along the fitted ellipse. Ellipse geometry errors are obtained from the errors in the
    coefficients of the 1st and 2nd simultaneous harmonic fit. 3rd and 4th harmonic amplitude errors
    are obtained in the same way, but only after the 1st and 2nd harmonics are subtracted from the
    raw data. See error analysis in Busko, I., 1996, Proceedings of the Fifth Astronomical Data
    Analysis Software and Systems Conference, Tucson, PASP Conference Series v.101, ed. G.H. Jacoby
    and J. Barnes, p.139-142.

    After fitting the ellipse that corresponds to a given value of the semi-major axis (by the
    process described above), the axis length is incremented/decremented following a pre-defined
    rule. At each step, the starting, first guess ellipse parameters are taken from the previously
    fitted ellipse that has the closest semi-major axis length to the current one. On low surface
    brightness regions (i.e., those having large radii), the small values of the image radial
    gradient can induce large corrections and meaningless values for the ellipse parameters. The
    algorithm has capabilities to stop increasing semi-major axis based on several criteria, including
    signal-to-noise ratio.

    See documentation of class Isophote for the meaning of the stop code reported after each fit.

    The fit algorithm provides a k-sigma clipping algorithm for cleaning deviant sample points at each
    isophote, thus improving convergency stability against any non-elliptical structure such as stars,
    spiral arms, HII regions, defects, etc.

    The fit algorithm has no way of finding where, in the input image frame, the galaxy to be measured
    sits in. The center X,Y coordinates need to be close to the actual center for the fit to work. An
    "object locator" function helps to verify that the selected position can be used as starting point.
    This function scans a 10 X 10 window centered either on the X,Y coordinates in the Geometry instance
    passed to the constructor of the Ellipse class, or, if any one of them, or both, are set to None,
    on the input image frame center. In case a successful acquisition takes place, the Geometry instance
    is modified in place to reflect the solution of the object locator algorithm.

    In some cases the object locator algorithm may fail, even though there is enough signal-to-noise
    to start a fit (e.g. in objects with very high ellipticity). In those cases the sensitivity of
    the algorithm can be decreased by decreasing the value of the object locator threshold parameter.
    The locator can be shut off entirely by setting the threshold to zero.

    A note of caution: the algorithm was designed explicitly with a (elliptical) galaxy brightness
    distribution in mind. In particular, a well defined negative radial intensity gradient across
    the region being fitted is paramount for the achievement of stable solutions. Use of the
    algorithm in other types of images (e.g., planetary nebulae) may lead to inability to converge
    to any acceptable solution.

    '''
    def __init__(self, image, geometry=None, threshold=DEFAULT_THRESHOLD):
        '''
        Constructor

        :param image: np 2-D array
            image array
        :param geometry: instance of Geometry
            the optional geometry that describes the first ellipse to be fitted.
            If None, a default Geometry instance centered on the image frame and
            with ellipticity 0.2 and position angle 90 deg. is created.
        :param threshold: float, default = 1.0
            Threshold for the object locator algorithm. By lowering this value
            the object locator becomes less strict, in the sense that it will
            accept lower signal-to-noise data. If set to zero, the locator is
            effectively shut off. In this case, either the geometry information
            supplied by the 'geometry' parameter is used as is, or the fit
            algorithm will terminate prematurely. Note that, once the object
            locator runs successfully, the X and Y coordinates in the geometry
            instance are modified for good.
        '''
        self.image = image

        if geometry:
            self._geometry = geometry
        else:
            _x0 = image.shape[0] / 2
            _y0 = image.shape[1] / 2

            self._geometry = Geometry(_x0, _y0, 10., DEFAULT_EPS, np.pi/2)

        # run object locator
        self._locator = Locator(image, self._geometry)
        self._locator.locate(threshold=threshold)

    def set_threshold(self, threshold):
        '''
        Modify the threshold value used by the locator.

        :param threshold: float
            the new threshold value to use
        '''
        self._locator.threshold = threshold

    def fit_image(self, sma0 = 10.,
                          minsma      = 0.,
                          maxsma      = None,
                          step        = DEFAULT_STEP,
                          conver      = DEFAULT_CONVERGENCY,
                          minit       = DEFAULT_MINIT,
                          maxit       = DEFAULT_MAXIT,
                          fflag       = DEFAULT_FFLAG,
                          maxgerr     = DEFAULT_MAXGERR,
                          sclip       = DEFAULT_SCLIP,
                          nclip       = 0,
                          integrmode  = BI_LINEAR,
                          linear      = False,
                          maxrit      = None,
                          verbose     = True):
        # This parameter list is quite large and should in principle be simplified
        # by re-distributing these controls to somewhere else. We keep this design
        # though because it better mimics the flat architecture used in the original
        # STSDAS task 'ellipse'.
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
            if 'linear=False'). See 'linear' parameter.
        :param conver: float, default = 0.05
            main convergency criterion. Iterations stop when the
            largest harmonic amplitude becomes smaller (in absolute
            value) than 'conver' times the harmonic fit rms.
        :param minit: int, default = 10
            minimum number of iterations to perform. A minimum of 10
            iterations guarantees that, on average, 2 iterations will
            be available for fitting each independent parameter (the
            four harmonic amplitudes and the intensity level). In the
            first isophote, the minimum number of iterations is 2 * 'minit',
            to ensure that, even departing from not-so-good initial values,
            the algorithm has a better chance to converge to a sensible
            solution.
        :param maxit: int, default = 50
            maximum number of iterations to perform
        :param fflag: float, default = 0.7
            acceptable fraction of flagged data points in sample.
            If the actual number of valid data points is smaller
            than this, stop iterating and return current Isophote.
            Flagged data points are points that either lie outside
            the image frame, or where rejected by sigma-clipping.
        :param maxgerr: float, default = 0.5
            maximum acceptable relative error in the local radial
            intensity gradient. This is the main control for preventing
            ellipses to grow to regions of too low signal-to-noise ratio.
            It specifies the maximum acceptable relative error in the
            local radial intensity gradient. Experiments (see paper
            quoted in the 'ellipse' help page) showed that the fitting
            precision relates to that relative error. The usual behavior
            of the gradient relative error is to increase with semi-major
            axis, being larger in outer, fainter regions of a galaxy
            image. In the current implementation, the 'maxgerr' criterion
            is triggered only when two consecutive isophotes exceed the
            value specified in the parameter. This prevents premature
            stopping caused by contamination such as stars and HII
            regions.
            A number of actions may happen when the current gradient
            error exceeds 'maxgerr' (or becomes non-significant and is
            set to None) in the process of increasing semi-major axis
            length. If the maximum semi-major axis specified by parameter
            'maxsma' is set to None, semi-major axis grow is stopped and
            the algorithm proceeds inwards to the galaxy image center. If
            'maxsma' is set to some finite value, and this value is larger
            than the current semi-major axis length, the algorithm enters
            non-iterative mode and proceeds outwards until reaching 'maxsma'.
        :param sclip: float, default = 3.0
            sigma-cliping criterion
        :param nclip: int, default = 0
            number of iterations in sigma-cliping algorithm.
            If zero, ignore sigma-clip.
        :param integrmode: string, default = 'bi-linear'
            area integration mode, as defined in module integrator.py
        :param linear: boolean, default False
            semi-major axis growing/shrinking mode. If False, geometric
            growing mode is chosen, thus the semi-major axis length is
            increased by a factor of (1.+'step'), and the process is repeated
            until either the semi-major axis value reaches the value of
            parameter 'maxsma', or the last fitted ellipse has more than a
            given fraction of its sampled points flagged out (see 'fflag').
            The process then resumes from the first fitted ellipse (at 'sma0')
            inwards, in steps of (1./(1.+'step')), until the semi- major axis
            length reaches the value 'minsma'. In case of linear growing, the
            increment or decrement value is given directly by 'step' in pixels.
            If 'maxsma' is set to None, the semi-major axis will grow until a
            low signal-to-noise criterion is met. See 'maxgerr'.
        :param maxrit: float, default None
            maximum value of semi-major axis to perform an actual fit.
            Whenever the current semi-major axis length is larger than
            maxrit, the isophotes wil be just extracted using the current
            geometry, without being fitted. Ignored if None.
            This non-iterative mode may be useful for sampling regions
            of very low surface brightness, where the algorithm may become
            unstable and unable to recover reliable geometry information.
            Non-iterative mode can also be entered automatically whenever
            the ellipticity exceeds 1.0 or the ellipse center crosses the
            image boundaries.
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
        first_isophote = True
        while True:

            # first isophote runs longer
            minit_a = 2 * minit if first_isophote else minit
            first_isophote = False

            isophote = self.fit_isophote(sma, step, conver, minit_a, maxit, fflag, maxgerr,
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
            main convergency criterion. Iterations stop when the
            largest harmonic amplitude becomes smaller (in absolute
            value) than 'conver' times the harmonic fit rms.
        :param minit: int, default = 10
            minimum number of iterations to perform. A minimum of 10
            iterations guarantees that, on average, 2 iterations will
            be available for fitting each independent parameter (the
            four harmonic amplitudes and the intensity level). In the
            first isophote, the minimum number of iterations is 2 * 'minit',
            to ensure that, even departing from not-so-good initial values,
            the algorithm has a better chance to converge to a sensible
            solution.
        :param maxit: int, default = 50
            maximum number of iterations to perform
        :param fflag: float, default = 0.7
            acceptable fraction of flagged data points in sample.
            If the actual number of valid data points is smaller
            than this, stop iterating and return current Isophote.
            Flagged data points are points that either lie outside
            the image frame, or where rejected by sigma-clipping.
        :param maxgerr: float, default = 0.5
            maximum acceptable relative error in the local radial
            intensity gradient. When fitting one isophote by itself,
            this parameter doesn't have any effect on the outcome.
        :param sclip: float, default = 3.0
            sigma-cliping criterion
        :param nclip: int, default = 0
            number of iterations in sigma-cliping algorithm.
            If zero, ignore sigma-clip.
        :param integrmode: string, default = 'bi-linear'
            area integration mode, as defined in module integrator.py
        :param linear: boolean, default = False
            semi-major axis growing/shrinking mode. When fitting just
            one isophote, this parameter is used only by the code that
            defines the details of how elliptical arc segments ("sectors")
            are extracted from the image, when using area extraction modes
            (see parameter 'integrmode')
        :param maxrit: float, default None
            maximum value of semi-major axis to perform an actual fit.
            If the passed 'sma' value is larger than 'maxrit', the
            isophote wil be just extracted using the current geometry,
            without being fitted. Ignored if None.
            This non-iterative mode may be useful for sampling regions
            of very low surface brightness, where the algorithm may become
            unstable and unable to recover reliable geometry information.
            Non-iterative mode can also be entered automatically whenever
            the ellipticity exceeds 1.0 or the ellipse center crosses the
            image boundaries.
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
            defines the sense of SMA growth. When fitting just one isophote,
            this parameter is used only by the code that defines the details
            of how elliptical arc segments ("sectors") are extracted from
            the image, when using area extraction modes (see parameter
            'integrmode')
        :param isophote_list: list, default = None
            fitted Isophote instance is appended to this list. Must
            be created and managed by the caller.
        :return: Isophote instance
            the fitted isophote. The fitted isophote is also appended
            to the input list passed via parameter 'isophote_list'.
        '''
        geometry = self._geometry

        # if available, geometry from last fitted isophote will be
        # used as initial guess for next isophote.
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


class Locator(object):
    '''
    Object locator.

    '''
    def __init__(self, image, geometry):
        '''
        Object locator.

        :param image: np 2-D array
            image array
        :param geometry: instance of Geometry
            geometry that directs the locator to look at its X/Y
            coordinates. These are modified by the locator algorithm.
        '''
        self._image = image
        self._geometry = geometry

        self.threshold = DEFAULT_THRESHOLD

        self._in_mask = [
            [0,0,0,0,0, 0,0,0,0,0],
            [0,0,0,0,0, 0,0,0,0,0],
            [0,0,0,0,1, 1,0,0,0,0],
            [0,0,0,1,0, 0,1,0,0,0],
            [0,0,1,0,0, 0,0,1,0,0],

            [0,0,1,0,0, 0,0,1,0,0],
            [0,0,0,1,0, 0,1,0,0,0],
            [0,0,0,0,1, 1,0,0,0,0],
            [0,0,0,0,0, 0,0,0,0,0],
            [0,0,0,0,0, 0,0,0,0,0],
        ]
        # self._in_mask = [
        #     [0,0,0,0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0,0],
        #     [0,0,0,1,1,1,1,0,0,0],
        #     [0,0,0,1,1,1,1,0,0,0],
        #     [0,0,0,1,1,1,1,0,0,0],
        #     [0,0,0,1,1,1,1,0,0,0],
        #     [0,0,0,0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0,0],
        # ]
        self._out_mask = [
            [0,0,0,1,1,1,1,0,0,0],
            [0,0,1,0,0,0,0,1,0,0],
            [0,1,0,0,0,0,0,0,1,0],
            [1,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,1],
            [0,1,0,0,0,0,0,0,1,0],
            [0,0,1,0,0,0,0,1,0,0],
            [0,0,0,1,1,1,1,0,0,0],
        ]

        self._in_mask_npix = np.sum(np.array(self._in_mask))
        self._out_mask_npix = np.sum(np.array(self._out_mask))

    def locate(self, threshold=DEFAULT_THRESHOLD):
        '''
        Runs the object locator, modifying in place the geometry
        associated with this Ellipse instance.

        :param threshold: float, default = 1.0
            object locator threshold. To turn off the locator, set this to zero.
        '''
        # Check if center coordinates point to somewhere inside the frame.
        # If not, set then to frame center.
        _x0 =  self._geometry.x0
        _y0 =  self._geometry.y0
        if _x0 is None or _x0 < 0 or _x0 >= self._image.shape[0] or \
           _y0 is None or _y0 < 0 or _y0 >= self._image.shape[1]:
            _x0 = self._image.shape[0] / 2
            _y0 = self._image.shape[1] / 2

        # 1/2 size of square window
        rwindow = len(self._in_mask) /2

        max_fom = 0.
        max_i = 0
        max_j = 0

        for i in range(int(_x0 - rwindow), int(_x0 + rwindow) + 1):
            for j in range(int(_y0 - rwindow), int(_y0 + rwindow) + 1):
                # Re-centering window.
                i1 = max(0, i - rwindow)
                j1 = max(0, j - rwindow)
                i2 = min(self._image.shape[0]-1, i + rwindow)
                j2 = min(self._image.shape[1]-1, j + rwindow)

                window = self._image[j1:j2,i1:i2]

                # averages in inner and outer regions.
                inner = ma.masked_array(window, mask=self._in_mask)
                outer = ma.masked_array(window, mask=self._out_mask)

                inner_sum = np.sum(inner) / self._in_mask_npix
                outer_sum = np.sum(outer) / self._out_mask_npix
                # inner_sum = np.sum(inner)
                # outer_sum = np.sum(outer)
                inner_std = np.std(inner)
                outer_std = np.std(outer)
                stddev = np.sqrt(inner_std**2 + outer_std**2)

                fom = (inner_sum - outer_sum) / stddev

                if fom > max_fom:
                    max_fom = fom
                    max_i = i
                    max_j = j

                print ('@@@@@@     line: 679  - ', fom, max_fom, i, j, " - ", i1, i2, j1, j2, " - ", inner_sum, outer_sum, (inner_sum-outer_sum))


        print ('@@@@@@     line: 698  - ', max_i, max_j)


# 68	        ic = 0
# 69	        jc = 0
# 70	        fom = 0.
# 71
# 72	        if (list) {
# 73	            call printf ("Running object locator... ")
# 74	            call flush (STDOUT)
# 75	        }
# 76
# 77	        # Scan window.
# 78	        do j = j1, j2 {
# 79	            do i = i1, i2 {
# 80
# 81	                # Extract two concentric circular samples.
# 82	                call el_get (im, sec, real(i), real(j), INNER_RADIUS, 0.0, 0.0,
# 83	                             bufx, bufy, nbuffer, NPOINT(is), NDATA(is),
# 84	                             mean1, std1, ASTEP(al), LINEAR(al),
# 85	                             INT_LINEAR, 4., 4.0, 0, SAREA(is))
# 86	                call el_get (im, sec, real(i), real(j), OUTER_RADIUS, 0.0, 0.0,
# 87	                             bufx, bufy, nbuffer, NPOINT(is), NDATA(is),
# 88	                             mean2, std2, ASTEP(al), LINEAR(al),
# 89	                             INT_LINEAR, 4., 4.0, 0, SAREA(is))
# 90
# 91	                # Figure of merit measures if there is reasonable
# 92	                # signal at position i,j.
# 93	                if (IS_INDEF(std1)) std1 = 0.0
# 94	                if (IS_INDEF(std2)) std2 = 0.0
# 95	                aux = std1 * std1 + std2 * std2
# 96
# 97
# 98
# 99	                if (aux > 0.0) {
# 100	                    aux = (mean1 - mean2) / sqrt(aux)
# 101	                    if (aux > fom) {
# 102	                        fom = aux
# 103	                        ic  = i
# 104	                        jc  = j
# 105	                    }
# 106	                }
# 107	            }
# 108	        }
# 109	        call mfree (bufy, TY_REAL)
# 110	        call mfree (bufx, TY_REAL)
# 111
# 112	        if (list)
# 113	            call printf ("Done.\n")
# 114
# 115	        # If valid object, re-center if asked for. Otherwise, no-detection
# 116	        # is signaled by setting center coordinates to INDEF.
# 117	        if (fom > thresh) {
# 118	            if (recenter) {
# 119	                XC(is) = real (ic)
# 120	                YC(is) = real (jc)
# 121	            }
# 122	        } else {
# 123	            XC(is) = INDEFR
# 124	            YC(is) = INDEFR
# 125	        }
# 126
# 127	        # Restore coordinates to physical system if needed.
# 128	        if ((!IS_INDEFR (XC(is))) && (!IS_INDEFR (YC(is)))) {
# 129	            if (PHYSICAL(is)) {
# 130	                XC(is) = el_s2p (im, XC(is), 1)
# 131	                YC(is) = el_s2p (im, YC(is), 2)
# 132	            }
# 133	        }
# 134
#
