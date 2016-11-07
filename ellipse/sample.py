from __future__ import division

import math

import numpy as np

import ellipse.integrator as I
from ellipse.integrator import integrators, BI_LINEAR

# limits for sector angular width
PHI_MAX = 0.2
PHI_MIN = 0.05


class Sample(object):

    def __init__(self, image, sma, x0=None, y0=None, astep=0.1, eps=0.2, position_angle=0.0,
                 linear=False, integrmode=BI_LINEAR):

        self.image = image
        self.astep = astep
        self.linear = linear
        self.integrmode = integrmode

        # Many parameters below can be made private.
        # Each integration method may need just a
        # subset of them. Later on, we should be
        # able to initialize only the ones needed
        # for the given integration mode. We should
        # also move whatever we can to local contexts,
        # minimizing the number of attributes in 'self'.

        # initialize ellipse scanning
        self.npoint = 0
        self.s      = 0.0
        self.s2     = 0.0
        self.aarea  = 0.0

        # if no center was specified, assume it's roughly
        # coincident with the image center
        _x0 = x0
        _y0 = y0
        if not _x0 or not _y0:
            _x0 = image.shape[0] / 2
            _y0 = image.shape[1] / 2

        self.geometry = Geometry(_x0, _y0, sma, eps, position_angle)

        # limiting annulus ellipses
        a1, a2 = I.limiting_ellipses(self.geometry.sma, self.astep, self.linear)

        self._inner_geometry = Geometry(_x0, _y0, a1, eps, position_angle)
        self._outer_geometry = Geometry(_x0, _y0, a2, eps, position_angle)

        # parameters for building first sector
        self.radius = sma
        aux         = min ((a2 - a1), 3.)
        self.sarea  = (a2 - a1) * aux
        self.dphi   = max (min ((aux / self.geometry.sma), PHI_MAX), PHI_MIN)
        self.phi    = self.dphi / 2.
        self.phi2   = self.phi - self.dphi / 2.
        aux         = 1. - self.geometry.eps
        self.r3     = a2 * aux / math.sqrt ((aux * math.cos (self.phi2))**2 + (math.sin (self.phi2))**2)
        self.r4     = a1 * aux / math.sqrt ((aux * math.cos (self.phi2))**2 + (math.sin (self.phi2))**2)

    def extract(self):
        ''' Build sample by scanning elliptical path over image array

            :return: 2-d array with three elements. Each element is a 1-d
                     array containing respectively angles, radii, and
                     extracted intensity values.
        '''
        # individual extracted sample points will be stored in here
        angles = []
        radii = []
        intensities = []

        # build integrator
        integrator = integrators[self.integrmode](self.image, self.geometry, angles, radii, intensities)

        # scan along elliptical path
        while (self.phi < np.pi*2.):

            integrator.integrate(self.radius, self.phi)

            # update angle and radius to be used to define
            # next sector along the elliptical path
            phistep_ = integrator.get_phi_step()
            self.phi += min (phistep_, 0.5)
            self.radius = self.geometry.sma * (1. - self.geometry.eps) / \
                          math.sqrt(((1. - self.geometry.eps) * math.cos(self.phi))**2 + (math.sin(self.phi))**2)

        # average sector area is probably calculated after the integrator had time to
        # step over the entire elliptical path. But this remains to be seen. It's not
        # needed for now anyhow.
        self._sector_area = integrator.get_sector_area()

        # pack results in 2-d array
        result = np.array([np.array(angles), np.array(radii), np.array(intensities)])

        return result


class Geometry(object):
    '''
    This is basically a container that allows storage of all parameters
    associated with a given ellipse's geometry. It will eventually be
    augmented with geometry-associated operations.
    '''
    def __init__(self, x0, y0, sma, eps, pa):
        '''

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
        '''
        self.x0  = x0
        self.y0  = y0
        self.sma = sma
        self.eps = eps
        self.pa  = pa


