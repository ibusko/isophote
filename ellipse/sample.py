from __future__ import division

import math

import numpy as np

# limits for sector angular width
PHI_MAX = 0.2
PHI_MIN = 0.05


class Sample(object):

    def __init__(self, image, sma, astep=0.1, eps=0.2, linear=False):

        self.image = image
        self.sma = sma

        self.eps = eps
        self.astep = astep
        self.linear = linear

        # initialize ellipse scanning
        self.npoint = 0
        self.s      = 0.0
        self.s2     = 0.0
        self.aarea  = 0.0

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

            :return: structured array with three columns: 'theta', 'radius', 'intensity'
        '''

        # individual extracted sample points will be stored in here
        angles = []
        radii = []
        intensities = []

        while (self.phi < np.pi*2.):









            # this depends on sampling method being used
            phistep = 2. / self.sma









            # store results for later packing in structured array
            angles.append(self.phi)
            radii.append(self.radius)
            intensities.append(0.0)

            # step along elliptical path by updating angle and radius
            # to be  used to define next sector
            self.phi += min (phistep, 0.5)
            self.radius = self.sma * (1. - self.eps) / math.sqrt (((1. - self.eps) * math.cos (self.phi))**2 + (math.sin (self.phi))**2)

        # pack results in structured array
        result = np.array([np.array(angles), np.array(radii), np.array(intensities)],
                          dtype=[('theta', 'f4'),('radius', 'f4'),('intensity', 'f4')])

        return result