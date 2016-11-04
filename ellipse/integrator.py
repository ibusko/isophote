from __future__ import division

import math

# integration modes
NEAREST_NEIGHBOR = 'nearest_neighbor'
BI_LINEAR = 'bi-linear'


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

    def _store_results(self, phi, radius, sample):
        self._angles.append(phi)
        self._radii.append(radius)
        self._intensities.append(sample)

    def get_phi_step(self):
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

    def get_phi_step(self):
        return 2. / self._r

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


                # print("@@@@@@  file integrator.py; line 120 - ", self._image[j][i], qx, qy )
                # print("@@@@@@  file integrator.py; line 121 - ", self._image[j+1][i], fx, qy )
                # print("@@@@@@  file integrator.py; line 122 - ", self._image[j][i+1], qx, fy )
                # print("@@@@@@  file integrator.py; line 123 - ", self._image[j+1][i+1], fx, fy )
                # print("@@@@@@  file integrator.py; line 124 - ", sample)
                # print("@@@@@@  file integrator.py; line 125 - ")



            else:
                sample = self._subpix (self._image, i, j, fx, fy)

            # store results
            self._store_results(phi, radius, sample)


#TODO investigate how this works. Need test case.

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

    def get_phi_step(self):
        return 1. / self._r

    def get_sector_area(self):
        return 2.


integrators = {
    NEAREST_NEIGHBOR: NearestNeighborIntegrator,
    BI_LINEAR: BiLinearIntegrator
}

