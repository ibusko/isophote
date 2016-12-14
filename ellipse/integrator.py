from __future__ import division

import math

# integration modes
NEAREST_NEIGHBOR = 'nearest_neighbor'
BI_LINEAR = 'bi-linear'
MEAN = 'mean'
MEDIAN = 'median'

DPHI_MAX = 0.2     # limits for sector angular width
DPHI_MIN = 0.01


class Integrator(object):

    def __init__(self, image, geometry, angles, radii, intensities):
        '''
        Constructor

        :param image: 2-d numpy array
             image array
        :param geometry: Geometry instance
            object that encapsulates geometry information about current ellipse
        :param angles: list
            output list; contains the angle values along the elliptical path
        :param radii:  list
            output list; contains the radius values along the elliptical path
        :param intensities: list
            output list; contains the extracted intensity values along the elliptical path
        '''
        self._image = image
        self._geometry = geometry

        self._angles = angles
        self._radii = radii
        self._intensities = intensities

        # for bounds checking
        self._i_range = range(self._image.shape[0])
        self._j_range = range(self._image.shape[1])

    def integrate(self, radius, phi):
        '''
        The three input lists are updated with one sample point taken
        from the image by a chosen integration method.

        Sub classes should implement the actual integration method.

        :param radius: float
            length of radius vector in pixels
        :param phi: float
            polar angle of radius vector
        '''
        raise NotImplementedError

    def _reset(self):
        '''
        Starts the results lists anew.

        This method is for internal use and shouldn't
        be used by external callers.
        '''
        self._angles = []
        self._radii = []
        self._intensities = []

    def _store_results(self, phi, radius, sample):
        self._angles.append(phi)
        self._radii.append(radius)
        self._intensities.append(sample)

    def get_polar_angle_step(self):
        raise NotImplementedError

    def get_sector_area(self):
        raise NotImplementedError


class NearestNeighborIntegrator(Integrator):

    def integrate(self, radius, phi):

        self._r = radius

        # Get image coordinates of (radius, phi) pixel
        i = int(radius * math.cos(phi + self._geometry.pa) + self._geometry.x0)
        j = int(radius * math.sin(phi + self._geometry.pa) + self._geometry.y0)

        # ignore data point if outside image boundaries
        if (i in self._i_range) and (j in self._j_range):

            # need to handle masked pixels here
            sample = self._image[j][i]

            # store results
            self._store_results(phi, radius, sample)

    def get_polar_angle_step(self):
        return max (min (2. / self._r, DPHI_MAX), DPHI_MIN)

    def get_sector_area(self):
        return 1.


# sqrt(number of cells) in target pixel
NCELL = 8

class BiLinearIntegrator(Integrator):

    def integrate(self, radius, phi):

        self._r = radius

        # Get image coordinates of (radius, phi) pixel
        x_ = radius * math.cos(phi + self._geometry.pa) + self._geometry.x0
        y_ = radius * math.sin(phi + self._geometry.pa) + self._geometry.y0
        i = int(x_)
        j = int(y_)
        fx = x_ - i
        fy = y_ - j

        # ignore data point if outside image boundaries
        if (i in self._i_range) and (j in self._j_range):

            # need to handle masked pixels here
            qx = 1. - fx
            qy = 1. - fy
            if (self._geometry.sma > 20.):
                sample = self._image[j][i]     * qx * qy + \
    	                 self._image[j+1][i]   * fx * qy + \
    	                 self._image[j][i+1]   * qx * fy + \
    	                 self._image[j+1][i+1] * fx * fy
            else:
                sample = self._subpix (self._image, i, j, fx, fy)

            # store results
            self._store_results(phi, radius, sample)

    def _subpix(self, image, i, j, fx, fy):

        z1 = image[j][i]
        z2 = image[j+1][i]
        z3 = image[j][i+1]
        z4 = image[j+1][i+1]

        sum = 0.
        a1  = z2 - z1
        a2  = z4 - z3
        a3  = 1./ NCELL
        correction = 0.5 + a3 / 2.
        for j in range(0, NCELL):
            y = j * a3 + fy - correction
            for i in range(0, NCELL):
                x = i * a3 + fx - correction
                za = a1 * x + z1
                zb = a2 * x + z3
                z  = (zb - za) * y + za
                sum = sum + z
        return sum / NCELL**2

    def get_polar_angle_step(self):
        return max (min (1. / self._r, DPHI_MAX), DPHI_MIN)

    def get_sector_area(self):
        return 2.


class AreaIntegrator(Integrator):

    def __init__(self, image, geometry, angles, radii, intensities):

        super(AreaIntegrator, self).__init__(image, geometry, angles, radii, intensities)

        # build auxiliary bi-linear integrator to be used when
        # sector areas contain a too small number of valid pixels.
        self._bi_linear_integrator = integrators[BI_LINEAR](image, geometry, angles, radii, intensities)

    def integrate(self, radius, phi):

        # Get image coordinates of the four vertices of the elliptical sector.
        vertex_x, vertex_y = self._geometry.initialize_sector_geometry(phi)

        self._sector_area = self._geometry.sector_area

        # step in polar angle to be used by caller next time
        # when updating the current polar angle 'phi' to point
        # to the next sector.
        self._phistep = self._geometry.sector_angular_width

        # define rectangular image area that
        # encompasses the elliptical sector.
        i1 = int(min(vertex_x))
        j1 = int(min(vertex_y))
        i2 = int(max(vertex_x))
        j2 = int(max(vertex_y))

        # polar angle limits for this sector
        phi1, phi2 = self._geometry.polar_angle_sector_limits()

        # ignore data point if the elliptical sector lies
        # partially, ou totally, outside image boundaries
        if (i1 in self._i_range) and (j1 in self._j_range) and \
           (i2 in self._i_range) and (j2 in self._j_range):

            # Scan rectangular image area, compute sample value.
            accumulator = self.initialize_accumulator()
            npix   = 0
            for j in range(j1,j2):
                for i in range(i1, i2):
                    # Check if polar coordinates of each pixel
                    # put it inside elliptical sector.
                    rp, phip = self._geometry.to_polar(i, j)

                    # check if inside angular limits
                    if phip < phi2 and phip >= phi1:

                        # check if radius is inside bounding ellipses
                        sma1, sma2 = self._geometry.bounding_ellipses()
                        aux = (1. - self._geometry.eps) / math.sqrt(((1. - self._geometry.eps) *
                              math.cos(phip))**2 + (math.sin(phip))**2)
                        sma1 *= aux
                        sma2 *= aux

                        if rp < sma2 and rp >= sma1:
                            # update accumulator with pixel value
                            pix_value = self._image[j][i]
                            accumulator, npix = self.accumulate(pix_value, npix, accumulator)

            # If 6 or less pixels were sampled, get the bi-linear interpolated value instead.
            if npix in range (1,7):
                # must reset integrator to remove older samples.
                self._bi_linear_integrator._reset()
                self._bi_linear_integrator.integrate(radius, phi)
                # because it was reset, current value is the only one stored
                # internally in the bi-linear integrator instance. Move it
                # from the internal integrator to this instance.
                sample_value = self._bi_linear_integrator._intensities[0]
                self._store_results(phi, radius, sample_value)

            elif npix > 6:
                sample_value = self.compute_sample_value(accumulator, npix)
                self._store_results(phi, radius, sample_value)

    def get_polar_angle_step(self):
        return max (min (self._phistep, DPHI_MAX), DPHI_MIN)

    def get_sector_area(self):
        return self._sector_area

    def initialize_accumulator(self):
        raise NotImplementedError

    def accumulate(self, pixel_value, npix, accumulator):
        raise NotImplementedError

    def compute_sample_value(self, accumulator, npix):
        raise NotImplementedError


class MeanIntegrator(AreaIntegrator):

    def initialize_accumulator(self):
        accumulator = 0.
        return accumulator

    def accumulate(self, pixel_value, npix, accumulator):
        accumulator += pixel_value
        npix += 1
        return accumulator, npix

    def compute_sample_value(self, accumulator, npix):
        return accumulator / npix


class MedianIntegrator(AreaIntegrator):

    def initialize_accumulator(self):
        accumulator = []
        return accumulator

    def accumulate(self, pixel_value, npix, accumulator):
        accumulator.append(pixel_value)
        npix += 1
        return accumulator, npix

    def compute_sample_value(self, accumulator, npix):
        accumulator.sort()
        return accumulator[int(npix/2)]


# Specific integrator subclasses can be instantiated from here.

integrators = {
    NEAREST_NEIGHBOR: NearestNeighborIntegrator,
    BI_LINEAR: BiLinearIntegrator,
    MEAN: MeanIntegrator,
    MEDIAN: MedianIntegrator
}

