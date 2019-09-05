"""model for the FDK neural network (baseline model)

This script is a model file
"""

import tensorflow as tf

import pyronn.ct_reconstruction.helpers.filters.weights as ct_weights
from pyronn.ct_reconstruction.layers.backprojection_3d import cone_backprojection3d
from pyronn.ct_reconstruction.helpers.filters import filters

from train_models import GEO_LIST


class ModelFDKNet:
    """
    The model class
    """

    def __init__(self, geometry):
        """
        Parameters
        ----------
        geometry : GeometryCone3D
           The geometry used for reconstruction
        """

        self.geometry = geometry

        self.cosine_weight = tf.get_variable(name='cosine_weight', dtype=tf.float32,
                                             initializer=ct_weights.cosine_weights_3d(geometry),
                                             trainable=True)

        self.recon_filter = tf.get_variable(name='recon_filter', dtype=tf.float32,
                                           initializer=filters.ram_lak_3D(geometry),
                                           trainable=True)

        self.relu_alpha = tf.get_variable(name='relu_alpha', shape=(1), dtype=tf.float32,
                                          initializer=tf.constant_initializer(0),
                                          trainable=True)

    def forward_proj_domain(self, sinogram):
        """
        the projection domain of the model for processing sinograms

        Parameters
        ----------
        sinogram : ndarray
            The projection data used for processing and reconstruction

        Returns
        -------
        ndarray
            the sinograms after processing
        """

        self.sinogram_cosine = tf.multiply(sinogram, self.cosine_weight)

        self.sinogram_fft = tf.fft(tf.cast(self.sinogram_cosine, dtype=tf.complex64))
        self.filtered_sinogram_fft = tf.multiply(self.sinogram_fft, tf.cast(self.recon_filter, dtype=tf.complex64))
        self.filtered_sinogram = tf.real(tf.ifft(self.filtered_sinogram_fft))

        return self.filtered_sinogram

    def forward_recon_domain(self, input, index):
        """
        the reconstruction domain of the model for reconstruction

        Parameters
        ----------
        input : ndarray
            The projection data used for reconstruction

        index : int
            The projection data are from which CT by the identification of the index

        Returns
        -------
        ndarray
            the reconstruted CT images after this model
        """

        self.reconstruction = tf.case({tf.equal(index, 1): lambda: cone_backprojection3d(input, GEO_LIST[0], hardware_interp=False),
                                     tf.equal(index, 2): lambda: cone_backprojection3d(input, GEO_LIST[1], hardware_interp=False),
                                     tf.equal(index, 3): lambda: cone_backprojection3d(input, GEO_LIST[2], hardware_interp=False),
                                     tf.equal(index, 4): lambda: cone_backprojection3d(input, GEO_LIST[3], hardware_interp=False),
                                     tf.equal(index, 5): lambda: cone_backprojection3d(input, GEO_LIST[4], hardware_interp=False),
                                     tf.equal(index, 6): lambda: cone_backprojection3d(input, GEO_LIST[5], hardware_interp=False),
                                     tf.equal(index, 7): lambda: cone_backprojection3d(input, GEO_LIST[6], hardware_interp=False),
                                     tf.equal(index, 8): lambda: cone_backprojection3d(input, GEO_LIST[7], hardware_interp=False),
                                     tf.equal(index, 9): lambda: cone_backprojection3d(input, GEO_LIST[8], hardware_interp=False),
                                     tf.equal(index, 10): lambda: cone_backprojection3d(input, GEO_LIST[9], hardware_interp=False),
                                     tf.equal(index, 11): lambda: cone_backprojection3d(input, GEO_LIST[10], hardware_interp=False),
                                     tf.equal(index, 12): lambda: cone_backprojection3d(input, GEO_LIST[11], hardware_interp=False),
                                     tf.equal(index, 13): lambda: cone_backprojection3d(input, GEO_LIST[12], hardware_interp=False),
                                     tf.equal(index, 14): lambda: cone_backprojection3d(input, GEO_LIST[13], hardware_interp=False),
                                     tf.equal(index, 15): lambda: cone_backprojection3d(input, GEO_LIST[14], hardware_interp=False),
                                     tf.equal(index, 16): lambda: cone_backprojection3d(input, GEO_LIST[15], hardware_interp=False),
                                     tf.equal(index, 17): lambda: cone_backprojection3d(input, GEO_LIST[16], hardware_interp=False),
                                     tf.equal(index, 18): lambda: cone_backprojection3d(input, GEO_LIST[17], hardware_interp=False),
                                     tf.equal(index, 19): lambda: cone_backprojection3d(input, GEO_LIST[18], hardware_interp=False),
                                     tf.equal(index, 20): lambda: cone_backprojection3d(input, GEO_LIST[19], hardware_interp=False)},
                                    default=(lambda: cone_backprojection3d(input, self.geometry, hardware_interp=False)), exclusive=True)

        self.recon_relu = self.para_relu(self.reconstruction)

        return self.recon_relu

    def para_relu(self, input):
        """
        parametric ReLU function

        Parameters
        ----------
        input : ndarray
            The input used for this activation function

        Returns
        -------
        ndarray
            the data after activation
        """

        pos = tf.nn.relu(input)
        neg = tf.multiply(tf.multiply(tf.subtract(input, tf.abs(input)), 0.5), self.relu_alpha)
        output_relu = tf.add(pos, neg)

        return output_relu