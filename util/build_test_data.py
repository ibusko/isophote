import math

import numpy as np

from ellipse.geometry import Geometry


def build(nx=512, ny=512, background=100., noise=1.E-6, i0=100., sma=40., eps=0.2, pa=0.):
    '''
    Builds artificial image for testing purposes

    :param nx: int
        image size
    :param ny: int
        image size
    :param background: float
        constant background level to be add to all pixels in the image
    :param noise: float
        standard deviation of the Gaussian noise to be added to all pixels in the image
    :param i0: float
        surface brightness over reference elliptical isophote
    :param sma: float
        semi-major axis length of reference elliptical isophote.
    :param eps: float
        ellipticity of reference isophote
    :param pa: float
        position angle of reference isophote
    :return: 2-d numpy array
        resulting image
    '''
    image = np.zeros((ny, nx)) + background

    x1 = nx/2
    y1 = ny/2

    g = Geometry(x1, y1, sma, eps, pa, 0.1, False)

    for j in range(ny):
        for i in range(nx):
            radius, angle = g.to_polar(i, j)
            e_radius = g.radius(angle)
            value = (lambda x: i0 * math.exp(-7.669 * ((x) ** 0.25 - 1.)))(radius / e_radius)
            image[j,i] += value

    # central pixel is messed up; replace it with interpolated value
    image[x1, y1] = (image[x1-1, y1] + image[x1+1, y1]) / 2

    image += np.random.normal(0., noise, image.shape)

    return image
