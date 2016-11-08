from __future__ import division

import math

from ellipse.geometry import PHI_MIN, PHI_MAX

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

    def get_phi_step(self):
        return 1. / self._r

    def get_sector_area(self):
        return 2.


class AreaIntegrator(Integrator):

    def integrate(self, radius, phi):

        self._r = radius

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

        print("@@@@@@  file integrator.py; line 177 - ",  i1, i2, j1, j2)












        i = int(radius * math.cos(phi + self._geometry.pa) + self._geometry.x0)
        j = int(radius * math.sin(phi + self._geometry.pa) + self._geometry.y0)
        if (i in self._i_range) and (j in self._j_range):
            sample = self._image[j][i]
            self._store_results(phi, radius, sample)




# 177	                # Read subraster. If outside image boundaries,
# 178	                # invalidate data point.
# 179	                if ((i1 > 0) && (i1 < IM_LEN (im, 1)) &&
# 180	                    (j1 > 0) && (j1 < IM_LEN (im, 2)) &&
# 181	                    (i2 > 0) && (i2 < IM_LEN (im, 1)) &&
# 182	                    (j2 > 0) && (j2 < IM_LEN (im, 2))) {
# 183	                    call el_getsec (im, sec, i1, i2, j1, j2)
# 184	                    # Create buffer for median computation.
# 185	                    if (integrmode == INT_MED)
# 186	                        call malloc (medbuf, (i2-i1+1)*(j2-j1+1) , TY_REAL)
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
# 219	                    }
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


    def get_phi_step(self):
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

