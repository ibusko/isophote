from __future__ import division

import math

import numpy as np

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
                     array containing respectively 'phi', 'radius', and
                     'intensity' values.
        '''
        # individual extracted sample points will be stored in here
        angles = []
        radii = []
        intensities = []

        # step in angle is coarser in nearest-neighbor mode.
        # sector area is unity in nearest-neighbor mode.
        # this should be re-defined when implementing other
        # integration modes.
        self._phistep = 2. / self.sma
        self._sector_area = 1.

        # scan along elliptical path
        while (self.phi < np.pi*2.):

            # support only nearest-neighbor integration for now.
            self._integrate_nearest_neighbor(angles, radii, intensities)

            # update angle and radius to be used to define
            # next sector along the elliptical path
            self.phi += min (self._phistep, 0.5)
            self.radius = self.sma * (1. - self.eps) / math.sqrt(((1. - self.eps) * math.cos(self.phi))**2 + (math.sin(self.phi))**2)

        # pack results in 2-d array
        result = np.array([np.array(angles), np.array(radii), np.array(intensities)])

        return result

    def _integrate_nearest_neighbor(self, angles, radii, intensities):
        #
        # The three input lists are updated with one sample point
        # taken from the image by nearest-neighbor integration.
        #

        # Get image coordinates of (radius, phi) pixel
        i = int(self.radius * math.cos(self.phi + self.position_angle) + self.x0)
        j = int(self.radius * math.sin(self.phi + self.position_angle) + self.y0)

        # ignore data point if outside image boundaries
        if ((i >= 0) and (i < self.image.shape[0]) and
            (j >= 0) and (j < self.image.shape[1])):

            # need to handle masked pixels here
            sample = self.image[j][i]

            # store results
            angles.append(self.phi)
            radii.append(self.radius)
            intensities.append(sample)

