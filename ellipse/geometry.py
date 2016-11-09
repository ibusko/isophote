from __future__ import division

import math

import numpy as np

# limits for sector angular width
PHI_MAX = 0.2
PHI_MIN = 0.05


# utility function used in the computation of elliptical sector areas.
def _area(sma, eps, phi, r):
    aux  = r * math.cos(phi) / sma
    signal = aux / abs(aux)
    if (abs(aux) >= 1.):
        aux = signal
    return (abs (sma ** 2 * (1.-eps)/2. * math.acos(aux)))


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

        The Geometry object also keeps track of *where* in the ellipse we are,
        when performing an 'extract' operation. This is mostly relevant when
        using an area integration mode (as opposed to a pixel integration mode)

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

    def initialize_sector_geometry(self, phi):
        '''
        Initialize geometry attributes associated to an elliptical sector at polar angle 'phi'.
        Computes:
         - the four vertices that define the elliptical sector on the pixel array.
         - sector area (in variable self.sector_area)
         - sector angular width

        :param phi: float
            polar angle (radians) where the sector is located.
        :return: tuple with two 1-D np arrays
            with the X and Y coordinates of each vertex.
        '''
        # These polar radii bound the region between the inner and
        # outer ellipses that define the first sector.
        eps_ = 1. - self.eps
        # polar vector at one side of the elliptical sector
        self.phi1 = phi - self.sector_angular_width / 2.
        r1 = self.sma1 * eps_ / math.sqrt((eps_ * math.cos(self.phi1))**2 + (math.sin(self.phi1))**2)
        r2 = self.sma2 * eps_ / math.sqrt((eps_ * math.cos(self.phi1))**2 + (math.sin(self.phi1))**2)
        # polar vector at the other side of the elliptical sector
        self.phi2 = phi + self.sector_angular_width / 2.
        r3 = self.sma2 * eps_ / math.sqrt((eps_ * math.cos(self.phi2))**2 + (math.sin(self.phi2))**2)
        r4 = self.sma1 * eps_ / math.sqrt((eps_ * math.cos(self.phi2))**2 + (math.sin(self.phi2))**2)

        # sector area
        sa1  = _area (self.sma1, self.eps, self.phi1, r1)
        sa2  = _area (self.sma2, self.eps, self.phi1, r2)
        sa3  = _area (self.sma2, self.eps, self.phi2, r3)
        sa4  = _area (self.sma1, self.eps, self.phi2, r4)
        self.sector_area = abs ((sa3 - sa2) - (sa4 - sa1))

        # angular width of sector. It is defined such that it
        # comes out with roughly constant area along the ellipse.
        self.sector_angular_width = max(min((self.sector_area / (r3 - r4) / r4), PHI_MAX), PHI_MIN)

        # compute the 4 vertices that define the elliptical sector.
        vertex_x = np.zeros(shape=4, dtype=float)
        vertex_y = np.zeros(shape=4, dtype=float)

        # vertices are labelled in counterclockwise sequence
        vertex_x[0] = r1 * math.cos (self.phi1 + self.pa) + self.x0
        vertex_y[0] = r1 * math.sin (self.phi1 + self.pa) + self.y0
        vertex_x[1] = r2 * math.cos (self.phi1 + self.pa) + self.x0
        vertex_y[1] = r2 * math.sin (self.phi1 + self.pa) + self.y0
        vertex_x[2] = r4 * math.cos (self.phi2 + self.pa) + self.x0
        vertex_y[2] = r4 * math.sin (self.phi2 + self.pa) + self.y0
        vertex_x[3] = r3 * math.cos (self.phi2 + self.pa) + self.x0
        vertex_y[3] = r3 * math.sin (self.phi2 + self.pa) + self.y0

        return vertex_x, vertex_y

    def _initialize(self):

        self.sma1, self.sma2 = self._bounding_ellipses()

        # this inner_sma_ variable has no particular significance
        # except that it is used to estimate an initial step along
        # the elliptical path. The actual step will be calculated
        # by the chosen area integration algorithm
        inner_sma_ = min((self.sma2 - self.sma1), 3.)
        self.sector_angular_width = max(min((inner_sma_ / self.sma), PHI_MAX), PHI_MIN)

        self.initial_polar_angle = self.sector_angular_width / 2.
        self.initial_polar_radius = self.radius(self.initial_polar_angle)

    def _bounding_ellipses(self):
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

    def to_polar(self, x, y):
        '''
        Given x,y coordinates on image grid, returns radius
        and polar angle on ellipse coordinate system. Takes
        care of different definitions for pa and phi:

        -PI < pa < PI
        0 < phi  < 2*PI

        Note that radius can be anything; solution is not tied
        to the semi-major axis length, but to the center position
        and tilt angle only.

        :param x: float
            image coordinate
        :param y: float
            image coordinate
        :return: 2 floats
            radius, angle
        '''
        x1 = x - self.x0
        y1 = y - self.y0

        radius = x1**2 + y1**2
        if (radius > 0.0):
            radius = math.sqrt(radius)
            angle  = math.asin(abs(y1) / radius)
        else:
            radius = 0.
            angle = 1.

        if x1 >= 0. and y1 < 0.:
            angle = 2*np.pi - angle
        elif x1 < 0. and y1 >= 0.:
            angle = np.pi - angle
        elif x1 < 0. and y1 < 0.:
            angle = np.pi + angle

        pa1 = self.pa
        if self.pa < 0.:
            pa1 = self.pa + 2*np.pi
        angle = angle - pa1
        if angle < 0.:
            angle = angle + 2*np.pi

        return radius, angle