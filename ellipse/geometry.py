from __future__ import division

import math

# limits for sector angular width
PHI_MAX = 0.2
PHI_MIN = 0.05


class Geometry(object):

    def __init__(self, x0, y0, sma, eps, pa, astep, linear_growth):
        '''
        This is basically a container that allows storage of all parameters
        associated with a given ellipse's geometry.

        Parameters that describe the relationship of a given ellipse with
        other associated ellipses are also encapsulated in this container.
        These associate ellipses may include e.g. the two (inner and outer)
        bounding ellipses that are used to build sectors along the elliptical
        path.

        :param x0: float
            center coordinate in pixels along image row
        :param y0: float
            center coordinate in pixels along image column
        :param sma: float
             semi-major axis in pixels
        :param eps: ellipticity
             ellipticity
        :param pa: float
             position angle of ellipse in relation
             to the +X axis of the image array.
        :param astep: float
            step value for growing/shrinking the semi-
            major axis. It can be expressed either in
            pixels (when 'linear_growth'=True) or in
            relative value (when 'linear_growth=False')
        :param linear_growth: boolean
            semi-major axis growing/shrinking mode
        '''
        self.x0  = x0
        self.y0  = y0
        self.sma = sma
        self.eps = eps
        self.pa  = pa

        self.astep = astep
        self.linear_growth = linear_growth

        self._initialize()

    def radius(self, angle):
        '''
        Given a polar angle, return the corresponding polar radius.

        :param angle: float
            polar angle (radians)
        :return: float
            polar radius (pixels)
        '''
        return self.sma * (1.-self.eps) / math.sqrt(((1.-self.eps) * math.cos(angle))**2 + (math.sin(angle))**2)

    def bounding_ellipses(self):
        '''
        Compute the semi-major axis of the two ellipses that bound
        the annulus where integrations take place.

        :return: tuple:
            with two floats - the smaller and larger values of
            SMA that define the annulus  bounding ellipses
        '''
        if (self.linear_growth):
            a1 = self.sma - self.astep / 2.
            a2 = self.sma + self.astep / 2.

        else:
            a1 = self.sma * (1. - self.astep/2.)
            a2 = self.sma * (1. + self.astep/2.)

        return a1, a2

    def _initialize(self):

        sma1_, sma2_ = self.bounding_ellipses()

        # this inner_sma_ variable has no particular significance
        # except that it is used to estimate an initial step along
        # the elliptical path. The actual step will be calculated
        # by the chosen area integration algorithm
        inner_sma_ = min((sma2_ - sma1_), 3.)
        dphi_ = max(min((inner_sma_ / self.sma), PHI_MAX), PHI_MIN)

        self.initial_polar_angle = dphi_ / 2.
        self.initial_polar_radius = self.radius(self.initial_polar_angle)

        # initial polar radii to the inner and outer bounding ellipses.
        # In the trivial case these will revert to values along the semi
        # major axis. We leave the generic calculation in here for
        # convenience.
        phi2_ = self.initial_polar_angle - dphi_ / 2.
        r3_ = sma2_ * (1.- self.eps) / math.sqrt(((1.- self.eps) * math.cos(phi2_))**2 + (math.sin(phi2_))**2)
        r4_ = sma1_ * (1.- self.eps) / math.sqrt(((1.- self.eps) * math.cos(phi2_))**2 + (math.sin(phi2_))**2)




