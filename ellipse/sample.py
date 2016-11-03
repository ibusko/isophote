from __future__ import division

import math

import numpy as np

from ellipse.integrator import NearestNeighborIntegrator

# limits for sector angular width
PHI_MAX = 0.2
PHI_MIN = 0.05


class Sample(object):

    def __init__(self, image, sma, x0=None, y0=None, astep=0.1, eps=0.2, position_angle=0.0, linear=False):

        self.image = image
        self.sma = sma

        self.eps = eps
        self.astep = astep
        self.position_angle = position_angle
        self.linear = linear

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

        self.x0 = x0
        self.y0 = y0
        # if no center was specified, assume it's roughly
        # coincident with the image center
        if not x0 or not y0:
            self.x0 = image.shape[0] / 2
            self.y0 = image.shape[1] / 2

        # limiting annulus ellipses
        if (self.linear):
            a1 = self.sma - self.astep / 2.
            a2 = self.sma + self.astep / 2.
        else:
            a1 = self.sma * (1. - ((1. - 1./self.astep) / 2.))
            a2 = self.sma * (1. + (self.astep - 1.) / 2.)

        # parameters for building first sector
        self.radius = sma
        aux         = min ((a2 - a1), 3.)
        self.sarea  = (a2 - a1) * aux
        self.dphi   = max (min ((aux / self.sma), PHI_MAX), PHI_MIN)
        self.phi    = self.dphi / 2.
        self.phi2   = self.phi - self.dphi / 2.
        aux         = 1. - self.eps
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

        # support only nearest-neighbor integration for now.
        integrator = NearestNeighborIntegrator(self.image, self.x0, self.y0, self.sma, self.position_angle, angles, radii, intensities)
        self._phistep = integrator.get_phi_step()

        # scan along elliptical path
        while (self.phi < np.pi*2.):

            integrator.integrate(self.radius, self.phi)

            # update angle and radius to be used to define
            # next sector along the elliptical path
            self.phi += min (self._phistep, 0.5)
            self.radius = self.sma * (1. - self.eps) / math.sqrt(((1. - self.eps) * math.cos(self.phi))**2 + (math.sin(self.phi))**2)

        # average sector area is probably calculated after the integrator had time to
        # step over the entire elliptical path. But this remains to be seen. It's not
        # needed for now anyhow.
        self._sector_area = integrator.get_sector_area()

        # pack results in 2-d array
        result = np.array([np.array(angles), np.array(radii), np.array(intensities)])

        return result


