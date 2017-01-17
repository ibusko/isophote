from __future__ import division

import copy

import numpy as np

from ellipse.geometry import Geometry
from ellipse.integrator import integrators, BI_LINEAR


class Sample(object):

    def __init__(self, image, sma, x0=None, y0=None, astep=0.1, eps=0.2, position_angle=0.0,
                 linear_growth=False, integrmode=BI_LINEAR, geometry=None):
        '''
        A Sample instance describes an elliptical path over the image, over which
        intensities can be extracted using a selection of integration algorithms.
        The Sample instance contains a 'geometry' attribute that describes its geometry.

        Parameters
        ----------
        :param image: numpy 2-d array
            pixels
        :param sma: float
            the semi-major axis length in pixels
        :param x0: float, default=None
            the X coordinate of the ellipse center
        :param y0: foat, default=None
            the Y coordinate of the ellipse center
        :param astep: float, default=0.1
            step value for growing/shrinking the semi-
            major axis. It can be expressed either in
            pixels (when 'linear_growth'=True) or in
            relative value (when 'linear_growth=False')
        :param eps: ellipticity, default=0.2
             ellipticity
        :param pa: float, default=0.0
             position angle of ellipse in relation
             to the +X axis of the image array.
        :param linear_growth: boolean, default=False
            semi-major axis growing/shrinking mode
        :param integrmode: string, default=BI_LINEAR
            algorithm used for area integration
        :param geometry: Geometry instance, default=None
            the geometry that describes the ellipse. This can be used in
            lieu of the explicit specification of parameters 'sma', 'x0',
            'y0', 'eps', etc. In any case, the Geometry instance
             becomes an attribute of the Sample object.

        Attributes
        ----------
        :param values: 2-d numpy array
            sampled values as a 2-d numpy array
            with the following structure:
            values[0] = 1-d array with angles
            values[1] = 1-d array with radii
            values[2] = 1-d array with intensity
        :param mean: float
            the mean intensity along the elliptical path
        :param gradient: float
            the local radial intensity gradient
        :param gradient_error: float
            the error associated with the local radial intensity gradient
        :param gradient_relative_error: float
            the relative error associated with the local radial intensity gradient
        :param sector_area: float
            the average area of the sectors along the
            elliptical path where the sample values
            were integrated from.
        :param total_points: int
            the total number of sample values that would
            cover the entire elliptical path
        :param actual_points: int
            the actual number of sample values that were
            taken from the image. It can be smaller than
            total_points when the ellipse encompasses
            regions outside the image.
        '''
        self.image = image
        self.integrmode = integrmode

        if geometry:
            # when the geometry is inherited from somewhere else,
            # its 'sma' attribute must be replaced by the value
            # explicitly passed to the constructor.
            self.geometry = copy.deepcopy(geometry)
            self.geometry.sma = sma
        else:
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
        self.gradient_error = None
        self.gradient_relative_error = None
        self.sector_area = None

        # total_points reports the total number of pairs angle-radius that
        # were attempted. actual_points reports the actual number of sampled
        # pairs angle-radius that resulted in valid values.
        self.total_points = 0
        self.actual_points = 0

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
        sector_areas = []

        # reset counters
        self.total_points = 0
        self.actual_points = 0

        # build integrator
        integrator = integrators[self.integrmode](self.image, self.geometry, angles, radii, intensities)

        # initialize walk along elliptical path
        radius = self.geometry.initial_polar_radius
        phi = self.geometry.initial_polar_angle

        # In case of an area integrator, ask the integrator to deliver a hint of how much
        # area the sectors will have. In case of too small areas, tests showed that the
        # area integrators (mean, median) won't perform properly. In that case, we override
        # the caller's selection and use the bi-linear integrator regardless.
        if integrator.is_area():
            integrator.integrate(radius, phi)
            area = integrator.get_sector_area()
            # this integration that just took place messes up with the storage
            # arrays and the constructors. We have to build a new integrator
            # instance from scratch, even if it is the same kind as originally
            # selected by the caller.
            angles = []
            radii = []
            intensities = []
            if area < 1.0:
                integrator = integrators[BI_LINEAR](self.image, self.geometry, angles, radii, intensities)
            else:
                integrator = integrators[self.integrmode](self.image, self.geometry, angles, radii, intensities)

        # walk along elliptical path, integrating at specified
        # places defined by polar vector.
        while (phi < np.pi*2.):

            # do the integration at phi-radius position, and append
            # results to the angles, radii, and intensities lists.
            integrator.integrate(radius, phi)

            # store sector area locally
            sector_areas.append(integrator.get_sector_area())

            # update total number of points
            self.total_points += 1

            # update angle and radius to be used to define
            # next polar vector along the elliptical path
            phistep_ = integrator.get_polar_angle_step()
            phi += min(phistep_, 0.5)
            radius = self.geometry.radius(phi)

        # average sector area is calculated after the integrator had
        # the opportunity to step over the entire elliptical path.
        self.sector_area = np.mean(np.array(sector_areas))

        # actual number of sampled points
        self.actual_points = len(angles)

        # pack results in 2-d array
        result = np.array([np.array(angles), np.array(radii), np.array(intensities)])

        return result

    def update(self, step=0.1):
        ''' Update this Sample instance with the mean intensity and
            local gradient values.

        :param step: float
            by how much to increment/decrement the semi-major axis in
            order to get a second sample that will enable estimation
            of the local gradient.
        '''
        # Update the mean value first, using extraction from main sample.
        s = self.extract()
        self.mean = np.mean(s[2])

        # Get sample with same geometry but at a different distance from
        # center. Estimate gradient from there.
        gradient, gradient_error = self._get_gradient(step)

        # Check for meaningful gradient. If no meaningful gradient, try
        # another sample, this time using larger radius. Meaningful
        # gradient means something  shallower, but still close to within
        # a factor 3 from previous gradient estimate. If no previous
        # estimate is available, guess it.
        previous_gradient = self.gradient
        if not previous_gradient:
            previous_gradient = -0.05 # good enough, based on usage

        if gradient >= (previous_gradient / 3.):   # gradient is negative!
            gradient, gradient_error = self._get_gradient(2 * step)

        # If still no meaningful gradient can be measured, try with previous
        # one, slightly shallower. A factor 0.8 is not too far from what is
        # expected from geometrical sampling steps of 10-20% and a
        # deVaucouleurs law or an exponential disk (at least at its inner parts,
        # r <~ 5 req). Gradient error is meaningless in this case.
        if gradient >= (previous_gradient / 3.):
            gradient = previous_gradient * 0.8
            gradient_error = None

        self.gradient = gradient
        self.gradient_error = gradient_error
        if gradient_error:
            self.gradient_relative_error = gradient_error / np.abs(gradient)
        else:
            self.gradient_relative_error = None

    def _get_gradient(self, step):
        gradient_sma = (1. + step) * self.geometry.sma

        gradient_sample = Sample(self.image, gradient_sma,
                                 x0=self.geometry.x0, y0=self.geometry.y0,
                                 astep=self.geometry.astep,
                                 eps=self.geometry.eps, position_angle=self.geometry.pa,
                                 linear_growth=self.geometry.linear_growth,
                                 integrmode=self.integrmode)

        sg = gradient_sample.extract()
        mean_g = np.mean(sg[2])
        gradient = (mean_g - self.mean) / self.geometry.sma / step

        s = self.extract()
        sigma = np.std(s[2])
        sigma_g = np.std(sg[2])

        gradient_error  = np.sqrt(sigma**2 / len(s[2]) + sigma_g**2 / len(sg[2])) / self.geometry.sma / step

        return gradient, gradient_error

    def coordinates(self):
        '''
        Returns the X-Y coordinates associated with each sampled point.

        :return: 1-D numpy arrays
            two arrays with the X and Y coordinates, respectively
        '''
        angles = self.values[0]
        radii = self.values[1]
        x = np.zeros(len(angles))
        y = np.zeros(len(angles))

        for i in range(len(x)):
            x[i] = radii[i] * np.cos (angles[i] + self.geometry.pa) + self.geometry.x0
            y[i] = radii[i] * np.sin (angles[i] + self.geometry.pa) + self.geometry.y0

        return x, y


class CentralSample(Sample):

    def update(self):
        ''' Overrides base class so as to update this Sample instance
            with the intensity integrated at the x0,y0 position using
            bi-linear integration. The local gradient is set to None.
        '''
        s = self.extract()
        self.mean = s[2][0]

        self.gradient = None
        self.gradient_error = None
        self.gradient_relative_error = None

    def _extract(self):
        angles = []
        radii = []
        intensities = []

        integrator = integrators[BI_LINEAR](self.image, self.geometry, angles, radii, intensities)
        integrator.integrate(0.0, 0.0)

        self.total_points = 1
        self.actual_points = 1

        return np.array([np.array(angles), np.array(radii), np.array(intensities)])
