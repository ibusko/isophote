from __future__ import division

import math

import numpy as np

# integration modes
NEAREST_NEIGHBOR = 'nearest_neighbor'
BI_LINEAR = 'bi-linear'
MEAN = 'mean'


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
        return 1. / self._r

    def get_sector_area(self):
        return 2.


class AreaIntegrator(Integrator):

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
        phi1 = self._geometry.phi1
        phi2 = self._geometry.phi2

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
                        aux = (1. - self._geometry.eps) / math.sqrt(((1. - self._geometry.eps) *
                              math.cos(phip))**2 + (math.sin(phip))**2)
                        rp1 = self._geometry.sma1 * aux
                        rp2 = self._geometry.sma2 * aux

                        if rp < rp2 and rp >= rp1:
                            # update accumulator with pixel value
                            pix_value = self._image[j][i]
                            accumulator, npix = self.accumulate(pix_value, npix, accumulator)

            if npix > 0:
                sample_value = self.compute_sample_value(accumulator, npix)
                self._store_results(phi, radius, sample_value)

            # Create buffer for median computation.
#           if (integrmode == INT_MED)
# 	            call malloc (medbuf, (i2-i1+1)*(j2-j1+1) , TY_REAL)
#           }
# 187	                    # Scan subraster, compute mean or median intensity.
# 188	                    sample = 0.
# 189	                    npix   = 0
# 190	                    do j = j1, j2 {
# 191	                        do i = i1, i2 {
# 192	                            # Check if polar coordinates of each subraster
# 193	                            # pixel put it inside elliptical sector.
# 194	                            call el_polar (float(i), float(j), x0, y0, teta,
# 195	                                           rp, phip)
# 196	                            if ((phip < phi2) && (phip >= phi1)) {
# 197	                                aux = (1. - eps) / sqrt (((1. - eps) *
# 198	                                      cos (phip))**2 + (sin (phip))**2)
# 199	                                rp1 = a1 * aux
# 200	                                rp2 = a2 * aux
# 201	                                if ((rp < rp2) && (rp >= rp1)) {
# 202	                                    pixel = Memr[SUBRASTER(sec) +
# 203	                                                 (j-j1) * (i2 - i1 + 1) +
# 204	                                                 i - i1]
# 205	                                    # Add valid pixel to accumulator.
# 206	                                    if (!IS_INDEFR (pixel)) {
# 207	                                        switch (integrmode) {
# 208	                                        case INT_MED:
# 209	                                            Memr[medbuf+npix] = pixel
# 210	                                            npix = npix + 1
# 211	                                        case INT_MEAN:
# 212	                                            accumulator = accumulator + pixel
# 213	                                            npix = npix + 1
# 214	                                        }
# 215	                                    }
# 216	                                }
# 217	                            }
# 218	                        }
# 220	                    # If 6 or less pixels were sampled, get the
# 221	                    # bi-linear interpolated value instead.
# 222	                    if (npix <= 6) {
# 223	                        r       = (r1 + r2 + r3 + r4) / 4.
# 224	                        area    = 2.
# 225	                        x[1]    = r * cos (phi + teta) + x0
# 226	                        y[1]    = r * sin (phi + teta) + y0
# 227	                        i       = x[1]
# 228	                        j       = y[1]
# 229	                        fx      = x[1] - real(i)
# 230	                        fy      = y[1] - real(j)
# 231	                        accumulator  = el_bilinear (im, sec, a, i, j, fx, fy)
# 232	                        if (integrmode == INT_MED)
# 233	                            call mfree (medbuf, TY_REAL)
# 234	                    } else {
# 235	                        # Compute mean or median.
# 236	                        switch (integrmode) {
# 237	                        case INT_MED:
# 238	                            call el_qsortr (Memr[medbuf], npix, el_comparer)
# 239	                            switch (mod (npix,2)) {
# 240	                            case 0:
# 241	                                accumulator = (Memr[medbuf + npix/2 - 1] +
# 242	                                          Memr[medbuf + npix/2]) / 2.
# 243	                            case 1:
# 244	                                accumulator = Memr[medbuf + npix/2]
# 245	                            }
# 246	                            call mfree (medbuf, TY_REAL)
# 247	                        case INT_MEAN:
# 248	                            accumulator = accumulator / float (npix)
# 249	                        }
# 250	                    }
# 251	                } else
# 252	                    accumulator = INDEFR
# 253


    def get_polar_angle_step(self):
        return self._phistep

    def get_sector_area(self):
        return self._sector_area

    def initialize_accumulator(self):
        raise NotImplementedError

    def accumulate(self, pixel_value, npix, sample):
        raise NotImplementedError

    def compute_sample_value(self, sample, npix):
        raise NotImplementedError


class MeanIntegrator(AreaIntegrator):

    def initialize_accumulator(self):
        sample = 0.
        return sample

    def accumulate(self, pixel_value, npix, sample):
        sample += pixel_value
        npix += 1
        return sample, npix

    def compute_sample_value(self, sample, npix):
        return sample / npix


integrators = {
    NEAREST_NEIGHBOR: NearestNeighborIntegrator,
    BI_LINEAR: BiLinearIntegrator,
    MEAN: MeanIntegrator
}

