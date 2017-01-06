from __future__ import (absolute_import, division, print_function, unicode_literals)

from ellipse.sample import Sample, CentralSample
from ellipse.fitter import Fitter, CentralFitter, TOO_MANY_FLAGGED
from ellipse.isophote import Isophote

from ellipse.integrator import BI_LINEAR

DEFAULT_STEP = 0.1
FIXED_ELLIPSE = 4
FAILED_FIT = 5


class Ellipse():
    '''
    This class provides the main access point to the isophote fitting algorithm.
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

    def fit_image(self, sma0=10., minsma=0., maxsma=None, step=DEFAULT_STEP,
                  integrmode=BI_LINEAR, linear=False, maxrit=None, verbose=False):
        '''
        Main fitting method. Fits multiple isophotes on the image array passed
        to the constructor. This method basically loops over each one of the
        values of semi-major axis length (sma) constructed from the input parameters,
        and fits one isophote at each sma, returning the entire set of isophotes in
        a sorted list.

        :param sma0: float, default = 10.
            starting value for the semi-major axis length (pixels)
        :param minsma:  float, default = 0.
            minimum value for the semi-major axis length (pixels)
        :param maxsma: float, default = None.
            maximum value for the semi-major axis length (pixels).
            When set to None, the algorithm will increase the semi
            major axis until one of several conditions will cause
            it to stop and revert to fit ellipses with sma < sma0.
        :param step: float, default = DEFAULT_STEP
            the step value being used to grow/shrink the semi-major
            axis length (pixels if 'linear=True', or relative value
            if 'linear=False')
        :param integrmode: string, default = BI_LINEAR
            integration mode, as defined in module integrator.py
        :param linear: boolean, default False
            semi-major axis growing/shrinking mode
        :param maxrit: float, default None
            maximum value of semi-major axis to perform an actual fit.
            Whenever the current semi-major axis length is larger than
            maxrit, the isophotes wil be just extracted using the current
            geometry, without being fitted. Ignored if None.
        :param verbose: boolean, default False
            print iteration info
        :return: list
            this list stores fitted Isophote instances, sorted according
            to the semi-major axis length value.
        '''

        # multiple fitted isophotes will be stored here
        isophote_list = []

        # first, go from initial sma outwards until
        # hitting one of several stopping criteria.
        sma = sma0
        noiter = False
        while True:
            isophote = self.fit_isophote(isophote_list, sma, step, integrmode, linear, maxrit, noniterate=noiter)

            # check for failed fit.
            if isophote.stop_code < 0 or isophote.stop_code == TOO_MANY_FLAGGED:

                # in case the fit failed right at the outset, return an empty
                # list. This is the usual case when the user provides initial
                # guesses that are too way off to enable the fitting algorithm
                # to find any meaningful solution.
                if len(isophote_list) == 1:
                    return []

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
            isophote = self.fit_isophote(isophote_list, sma, step, integrmode, linear, maxrit, going_inwards=True)

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
            isophote = self.fit_isophote(isophote_list, 0.0, step, integrmode, linear)
            isophote.print(verbose)

        # sort list of isophotes according to sma
        isophote_list.sort()

        return isophote_list

    def fit_isophote(self, isophote_list, sma, step=DEFAULT_STEP, integrmode=BI_LINEAR,
                     linear=False, maxrit=None, noniterate=False, going_inwards=False):
        '''
        Fit one isophote with a given semi-major axis length.

        The 'step' and 'linear' parameters are not used to actually
        grow or shrink the current fitting semi-major axis length.
        They are necessary nevertheless, so the sampling algorithm
        can know where to start the gradient computation, and also
        how to compute the elliptical sector areas (when area
        integration mode is selected).

        :param isophote_list: list
            fitted Isophote instance is appended to this list. Must
            be created and managed by the caller.
        :param sma: float
            the semi-major axis length (pixels)
        :param step: float, default = DEFAULT_STEP
            the step value being used to grow/shrink the semi-major
            axis length (pixels)
        :param integrmode: string, default = BI_LINEAR
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
        :return: Isophote instance
            the fitted isophote. The fitted isophote is also appended
            to the input list passed via parameter 'isophote_list'.
        '''
        # if available, geometry from last fitted isophote will be
        # used as initial guess for next isophote.
        geometry = self._geometry
        if len(isophote_list) > 0:
            geometry = isophote_list[-1].sample.geometry

        # do the fit.
        if noniterate or (maxrit and sma > maxrit):
            isophote = self._non_iterative(sma, step, linear, geometry)
        else:
            isophote = self._iterative(sma, step, linear, geometry, integrmode, going_inwards)

        # store result in list
        if isophote.valid:
            isophote_list.append(isophote)
        # For now, to facilitate regression comparisons, we
        # store all, and not just valid, isophotes in the list.
        # isophote_list.append(isophote)

        return isophote

    def _iterative(self, sma, step, linear, geometry, integrmode, going_inwards=False):
        if sma > 0.:
            # iterative fitter
            sample = Sample(self.image, sma,
                            astep=step,
                            linear_growth=linear,
                            geometry=geometry,
                            integrmode=integrmode)
            fitter = Fitter(sample)
        else:
            # sma == 0 requires special handling.
            sample = CentralSample(self.image, 0.0, geometry=geometry)
            fitter = CentralFitter(sample)

        isophote = fitter.fit(going_inwards=going_inwards)

        return isophote

    def _non_iterative(self, sma, step, linear, geometry):
        sample = Sample(self.image, sma,
                        astep=step,
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




