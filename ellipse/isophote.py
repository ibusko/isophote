from __future__ import (absolute_import, division, print_function, unicode_literals)


class Isophote:

    def __init__(self, sample, iter, valid):
        '''
        For now, this is just a container that helps in segregating information
        directly related to the sample (sampled intensities along an elliptical
        path on the image), from isophote-specific information.

        :param sample: instance of Sample
            the sample information
        :param iter: int
            number of iterations used to fit the isophote
        :param valid: boolean
            status of the fitting operation
        '''
        self.sample = sample
        self.iter = iter
        self.valid = valid

