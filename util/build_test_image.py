import pyfits
from scipy.ndimage import gaussian_filter
import numpy as np

nx, ny = 200, 200
image = np.zeros((ny, nx)) + 100.

x = nx / 2
y = ny / 2

image[y, x] += 5000.

image = gaussian_filter(image, (70,100))

image += np.random.normal(3., 0.02, image.shape)

pyfits.writeto('../test/synthetic_image_2.fits', image, clobber=True)