from __future__ import division

import numpy as np

from ellipse.geometry import Geometry
from ellipse.integrator import integrators, BI_LINEAR


class Sample(object):

    def __init__(self, image, sma, x0=None, y0=None, astep=0.1, eps=0.2, position_angle=0.0,
                 linear_growth=False, integrmode=BI_LINEAR):

        self.image = image
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

        self.geometry = Geometry(_x0, _y0, sma, eps, position_angle, astep,linear_growth)

    def extract(self):
        ''' Build sample by scanning elliptical path over image array

            :return: numpy 2-d array
                contains three elements. Each element is a 1-d
                array containing respectively angles, radii, and
                extracted intensity values.
        '''
        # individual extracted sample points will be stored in here
        angles = []
        radii = []
        intensities = []

        # build integrator
        integrator = integrators[self.integrmode](self.image, self.geometry, angles, radii, intensities)

        # initialize walk along elliptical path
        self.radius = self.geometry.initial_polar_radius
        self.phi = self.geometry.initial_polar_angle

        # walk along elliptical path, integrating at specified
        # places defined by polar vector.
        while (self.phi < np.pi*2.):

            integrator.integrate(self.radius, self.phi)

            # update angle and radius to be used to define
            # next polar vector along the elliptical path
            phistep_ = integrator.get_phi_step()
            self.phi += min (phistep_, 0.5)
            self.radius = self.geometry.radius(self.phi)

        # average sector area is probably calculated after the integrator had time to
        # step over the entire elliptical path. But this remains to be seen. It's not
        # needed for now anyhow.
        self._sector_area = integrator.get_sector_area()

        # pack results in 2-d array
        result = np.array([np.array(angles), np.array(radii), np.array(intensities)])

        return result

