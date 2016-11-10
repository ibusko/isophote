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

        i1 = int(vertex_x[0] - 1)
        j1 = int(vertex_y[0] - 1)
        i2 = int(vertex_x[3] + 1)
        j2 = int(vertex_y[3] + 1)

        phi1 = self._geometry.phi1
        phi2 = self._geometry.phi2


        print ('@@@@@@     line: 179  -     indices    ', i1, j1, i2, j2)

        # print ('@@@@@@     line: 191  - ',
        # (i1 in self._i_range) and (j1 in self._j_range) and \
        # (i2 in self._i_range) and (j2 i0n self._j_range)
        # )

        # ignore data point if the elliptical sector lies
        # partially, ou totally, outside image boundaries
        if (i1 in self._i_range) and (j1 in self._j_range) and \
           (i2 in self._i_range) and (j2 in self._j_range):

            print ('@@@@@@     line: 201  - ')

            # Scan sector, compute mean or median intensity.
            sample = 0.
            npix   = 0
            for j in range(j1,j2):

                print ('@@@@@@     line: 198  -    looping in j  ', j)

                for i in range(i1, i2):

                    print ('@@@@@@     line: 202  -    looping in i   ', i)

                    # Check if polar coordinates of each pixel
                    # put it inside elliptical sector.
                    rp, phip = self._geometry.to_polar(i, j)

                    print ('@@@@@@     line: 194  -   angle       ', phi1/np.pi*180, phip/np.pi*180, phi2/np.pi*180)

                    if phip < phi2 and phip >= phi1:

                        aux = (1. - self._geometry.eps) / math.sqrt(((1. - self._geometry.eps) *
                              math.cos(phip))**2 + (math.sin(phip))**2)
                        rp1 = self._geometry.sma1 * aux
                        rp2 = self._geometry.sma2 * aux

                        print ('@@@@@@     line: 201  -    radii      ',rp1, rp, rp2)

                        if rp < rp2 and rp >= rp1:
                            sample += self._image[j][i]
                            npix += 1

                            print ('@@@@@@     line: 207      npix    - ', npix)

            if npix > 0:
                self._store_results(phi, radius, sample / npix)

            print ('@@@@@@     line: 205  - ', phi/np.pi*180, radius, npix)

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
# 205	                                    # Add valid pixel to sample.
# 206	                                    if (!IS_INDEFR (pixel)) {
# 207	                                        switch (integrmode) {
# 208	                                        case INT_MED:
# 209	                                            Memr[medbuf+npix] = pixel
# 210	                                            npix = npix + 1
# 211	                                        case INT_MEAN:
# 212	                                            sample = sample + pixel
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
# 231	                        sample  = el_bilinear (im, sec, a, i, j, fx, fy)
# 232	                        if (integrmode == INT_MED)
# 233	                            call mfree (medbuf, TY_REAL)
# 234	                    } else {
# 235	                        # Compute mean or median.
# 236	                        switch (integrmode) {
# 237	                        case INT_MED:
# 238	                            call el_qsortr (Memr[medbuf], npix, el_comparer)
# 239	                            switch (mod (npix,2)) {
# 240	                            case 0:
# 241	                                sample = (Memr[medbuf + npix/2 - 1] +
# 242	                                          Memr[medbuf + npix/2]) / 2.
# 243	                            case 1:
# 244	                                sample = Memr[medbuf + npix/2]
# 245	                            }
# 246	                            call mfree (medbuf, TY_REAL)
# 247	                        case INT_MEAN:
# 248	                            sample = sample / float (npix)
# 249	                        }
# 250	                    }
# 251	                } else
# 252	                    sample = INDEFR
# 253


    def get_polar_angle_step(self):
        return self._phistep

    def get_sector_area(self):
        return self._sector_area


class MeanIntegrator(AreaIntegrator):
    pass





integrators = {
    NEAREST_NEIGHBOR: NearestNeighborIntegrator,
    BI_LINEAR: BiLinearIntegrator,
    MEAN: MeanIntegrator
}

