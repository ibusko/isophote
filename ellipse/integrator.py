from __future__ import division

import math


class Integrator(object):

    def __init__(self, image, sma, position_angle, angles, radii, intensities):
        '''
        Constructor

        :param image: 2-d numpy array
             image array
        :param sma: float
             semi-major axis in pixels
        :param position_angle: float
             position angle of ellipse in relation
             to the +X axis of the image array.
        :param angles: list
            output list; contains the angle values along the elliptical path
        :param radii:  list
            output list; contains the radius values along the elliptical path
        :param intensities: list
            output list; contains the extracted intensity values along the elliptical path
        '''
        self._image = image
        self._sma = sma
        self._position_angle = position_angle

        self._angles = angles
        self._radii = radii
        self._intensities = intensities

    def integrate(self, x0, y0, radius, phi):
        '''
        The three input lists are updated with one sample point taken
        from the image by a chosen integration method.

        Sub classes should implement the actual integration method.

        :param x0: float
            center coordinate in pixels along image row
        :param y0: float
            center coordinate in pixels along image column
        :param radius: float
            length of radius vector in pixels
        :param phi: float
            polar angle of radius vector
        '''
        raise NotImplementedError

    def get_phi_step(self):
        raise NotImplementedError

    def get_sector_area(self):
        raise NotImplementedError

class NearestNeighborIntegrator(Integrator):

    def integrate(self, x0, y0, radius, phi):

        # Get image coordinates of (radius, phi) pixel
        i = int(radius * math.cos(phi + self._position_angle) + x0)
        j = int(radius * math.sin(phi + self._position_angle) + y0)

        # ignore data point if outside image boundaries
        if ((i >= 0) and (i < self._image.shape[0]) and
            (j >= 0) and (j < self._image.shape[1])):

            # need to handle masked pixels here
            sample = self._image[j][i]

            # store results
            self._angles.append(phi)
            self._radii.append(radius)
            self._intensities.append(sample)

    def get_phi_step(self):
        return 2. / self._sma

    def get_sector_area(self):
        return 1.
