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

        self.geometry = Geometry(_x0, _y0, sma, eps, position_angle, astep, linear_growth)

        # extracted values associated with this sample.
        self.values = None
        self.mean = None
        self.gradient = None

    def extract(self):
        ''' Build sample by scanning elliptical path over image array

            :return: numpy 2-d array
                contains three elements. Each element is a 1-d
                array containing respectively angles, radii, and
                extracted intensity values.
        '''
        # the sample values themselves are kept cached to prevent
        # multiple calls to the integrator code.
        if self.values is not None:
            return self.values
        else:
            s = self._extract()
            self.values = s
            return s

    def _extract(self):
        # Here the actual sampling takes place. This is called only once
        # during the life of a Sample instance, because it's an expensive
        # calculation. This method should not be called from external code.
        # If one wants to force it to re-run, then do:
        #
        #   sample.values = None
        #
        # before calling sample.extract()

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
            phistep_ = integrator.get_polar_angle_step()
            self.phi += min (phistep_, 0.5)
            self.radius = self.geometry.radius(self.phi)

        # average sector area is probably calculated after the integrator had time to
        # step over the entire elliptical path. But this remains to be seen. It's not
        # needed for now anyhow.
        self._sector_area = integrator.get_sector_area()

        # pack results in 2-d array
        result = np.array([np.array(angles), np.array(radii), np.array(intensities)])

        return result

    def update(self, step=0.8):
        ''' Update this Sample instance with the mean intensity and
            local gradient values.

            Later we will add the mean and gradient errors as well.

        :param step: float
            by how much to increment/decrement the semi-major axis in
            order to get a second sample that will enable estimation
            of the local gradient.
        '''

        # Get sample with same geometry but at a different distance from center.
        gradient_sma = (1. + step) * self.geometry.sma

        gradient_sample = Sample(self.image, gradient_sma,
                                 x0=self.geometry.x0, y0=self.geometry.y0,
                                 astep=self.geometry.astep,
                                 eps=self.geometry.eps, position_angle=self.geometry.pa,
                                 linear_growth=self.geometry.linear_growth,
                                 integrmode=self.integrmode)

        print ('@@@@@@     line: 129  -   building sample for sma  ', self.geometry.sma)

        s  = self.extract()

        print ('@@@@@@     line: 133  - ', s[1])

        print ('@@@@@@     line: 133  -   building sample for gradient sma  ', gradient_sma)

        sg = gradient_sample.extract()

        print ('@@@@@@     line: 139  - ', sg[1])

        self.mean = np.mean(s[1])
        mean_g = np.mean(sg[1])

        self.gradient = (mean_g - self.mean) / self.geometry.sma / step

