"""model for using U-Net for slice-to-slice training

This script is a model file
"""

import tensorflow as tf

import pyronn.ct_reconstruction.helpers.filters.weights as ct_weights
from pyronn.ct_reconstruction.layers.backprojection_3d import cone_backprojection3d
from pyronn.ct_reconstruction.helpers.filters import filters

from train_models import GEO_LIST


class ModelProposedNet:
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

        self.initializer = tf.contrib.layers.xavier_initializer()

        self.cosine_weight = tf.get_variable(name='cosine_weight', dtype=tf.float32,
                                             initializer=ct_weights.cosine_weights_3d(geometry),
                                             trainable=True)

        self.recon_filter = tf.get_variable(name='recon_filter', dtype=tf.float32,
                                           initializer=filters.ram_lak_3D(geometry),
                                           trainable=True)

        self.relu_alpha = tf.get_variable(name='relu_alpha', shape=(1), dtype=tf.float32,
                                          initializer=tf.constant_initializer(0),
                                          trainable=True)

        # for multi perceptron if needed
        # self.mlp_all = tf.get_variable(name='mlp_all', dtype=tf.float32, initializer=tf.ones(1), trainable=False)
        # self.mlp_one = tf.get_variable(name='mlp_one', dtype=tf.float32, initializer=tf.ones(1), trainable=False)
        # self.mlp_two = tf.get_variable(name='mlp_two', dtype=tf.float32, initializer=tf.ones(1), trainable=False)

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

        self.weighted_sinogram_fft = tf.fft(tf.cast(self.sinogram_cosine, dtype=tf.complex64))
        self.filtered_sinogram_fft = tf.multiply(self.weighted_sinogram_fft, tf.cast(self.recon_filter, dtype=tf.complex64))
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
                                        default=(lambda:cone_backprojection3d(input, self.geometry, hardware_interp=False)), exclusive=True)

        self.recon_relu = self.para_relu(self.reconstruction)

        return self.recon_relu

    def unet_model(self, input):
        """
        the U-Net architecture defined for slice-to-slice training

        Parameters
        ----------
        input : ndarray
            The data used for this U-Net for feature mapping

        Returns
        -------
        ndarray
            the data after this U-Net feature mapping
        """

        input = tf.expand_dims(input, 3)

        with tf.variable_scope('reconstruction_unet', reuse=tf.AUTO_REUSE):
            # define downsample path
            h_conv = self.conv3x3_relu_layer(input, 1, 16, 'lev1_layer1')
            skip_con1 = self.conv3x3_relu_layer(h_conv, 16, 16, 'lev1_layer2')

            h_pool = self.max_pooling_layer(skip_con1, 'lev2_layer1')
            h_conv = self.conv3x3_relu_layer(h_pool, 16, 32, 'lev2_layer2')
            skip_con2 = self.conv3x3_relu_layer(h_conv, 32, 32, 'lev2_layer3')

            h_pool = self.max_pooling_layer(skip_con2, 'lev3_layer1')
            h_conv = self.conv3x3_relu_layer(h_pool, 32, 64, 'lev3_layer2')
            skip_con3 = self.conv3x3_relu_layer(h_conv, 64, 64, 'lev3_layer3')

            h_pool = self.max_pooling_layer(skip_con3, 'lev4_layer1')
            h_conv = self.conv3x3_relu_layer(h_pool, 64, 128, 'lev4_layer2')
            skip_con4 = self.conv3x3_relu_layer(h_conv, 128, 128, 'lev4_layer3')

            # the bottom
            h_pool = self.max_pooling_layer(skip_con4, 'lev5_layer1')
            h_conv = self.conv3x3_relu_layer(h_pool, 128, 256, 'lev5_layer2')
            h_conv = self.conv3x3_relu_layer(h_conv, 256, 256, 'lev5_layer3')

            # define upsample path
            h_conv = self.up_conv(h_conv, 256, 128, 'lev6_layer1')
            h_concat = self.copy_crop(skip_con4, h_conv)
            h_conv = self.conv3x3_relu_layer(h_concat, 256, 128, 'lev6_layer2')
            h_conv = self.conv3x3_relu_layer(h_conv, 128, 128, 'lev6_layer3')

            h_conv = self.up_conv(h_conv, 128, 64, 'lev7_layer1')
            h_concat = self.copy_crop(skip_con3, h_conv)
            h_conv = self.conv3x3_relu_layer(h_concat, 128, 64, 'lev7_layer2')
            h_conv = self.conv3x3_relu_layer(h_conv, 64, 64, 'lev7_layer3')

            h_conv = self.up_conv(h_conv, 64, 32, 'lev8_layer1')
            h_concat = self.copy_crop(skip_con2, h_conv)
            h_conv = self.conv3x3_relu_layer(h_concat, 64, 32, 'lev8_layer2')
            h_conv = self.conv3x3_relu_layer(h_conv, 32, 32, 'lev8_layer3')

            h_conv = self.up_conv(h_conv, 32, 16, 'lev9_layer1')
            h_concat = self.copy_crop(skip_con1, h_conv)
            h_conv = self.conv3x3_relu_layer(h_concat, 32, 16, 'lev9_layer2')
            h_conv = self.conv3x3_relu_layer(h_conv, 16, 16, 'lev9_layer3')

            # final output
            h_output = self.conv1x1(h_conv, 16, 1, 'lev9_layer4')

            h_output = tf.squeeze(h_output, axis=3)

            return h_output

    def conv3x3_relu_layer(self, input, in_channels, num_filters, name):
        """
        the convolutional layer with kernel size 3x3

        Parameters
        ----------
        input : ndarray
            The input used for this convolutional feature mapping

        in_channels : int
            The number of channels of the input

        num_filters : int
            The number of channels to map

        name : str
            The scope name

        Returns
        -------
        ndarray
            the data after doing this convolutional feature mapping
        """

        w_conv = tf.Variable(self.initializer([3,3,in_channels,num_filters]), name="{}_conv3x3_weight".format(name))
        b_conv = tf.Variable(self.initializer([num_filters]), name="{}_conv3x3_bias".format(name))
        h_conv = tf.nn.relu(tf.nn.conv2d(input, w_conv, strides=[1,1,1,1], padding='SAME', name="{}_conv3x3".format(name)) + b_conv,
                            name="{}_conv3x3_relu".format(name))

        return h_conv

    def max_pooling_layer(self, input, name):
        """
        the max pooling layer

        Parameters
        ----------
        input : ndarray
            The input used to do max pooling

        name : str
            The scope name

        Returns
        -------
        ndarray
            the data after max pooling
        """

        h_pool = tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="{}_max_pool".format(name))

        return h_pool

    def up_conv(self, input, in_channels, num_filters, name):
        """
        the upper convolutional layer

        Parameters
        ----------
        input : ndarray
            The input used for this convolutional feature mapping

        in_channels : int
            The number of channels of the input

        num_filters : int
            The number of channels to map

        name : str
            The scope name

        Returns
        -------
        ndarray
            the data after doing this upper convolutional feature mapping
        """

        w_upconv = tf.Variable(self.initializer([2,2,num_filters,in_channels]), name="{}_upconv_weight".format(name))
        b_upconv = tf.Variable(self.initializer([num_filters]), name="{}_upconv_bias".format(name))
        h_upconv = tf.nn.relu(tf.nn.conv2d_transpose(input, w_upconv, [tf.shape(input)[0], tf.shape(input)[1]*2, tf.shape(input)[2]*2, num_filters],
                                                      strides=[1,2,2,1], padding='SAME', name="{}_upconv".format(name)) + b_upconv,
                                                      name="{}_upconv_relu".format(name))

        return h_upconv

    def copy_crop(self, skip_connect, input):
        """
        the copy concatenation function for feature mappings (no crop for my project)

        Parameters
        ----------
        skip_connect : ndarray
            The first data to concatenate

        input : ndarray
            The first data to concatenate

        Returns
        -------
        ndarray
            the concatenated data for the two inputs
        """

        concat = tf.concat([skip_connect, input], axis=3)

        return concat

    def conv1x1(self, input, in_channels, num_filters, name):
        """
        the convolutional layer with kernel size 1x1

        Parameters
        ----------
        input : ndarray
            The input used for this convolutional feature mapping

        in_channels : int
            The number of channels of the input

        num_filters : int
            The number of channels to map

        name : str
            The scope name

        Returns
        -------
        ndarray
            the data after doing this convolutional feature mapping
        """

        w_conv = tf.Variable(self.initializer([1, 1, in_channels, num_filters]),
                             name="{}_conv1x1_weight".format(name))
        b_conv = tf.Variable(self.initializer([num_filters]), name="{}_conv1x1_bias".format(name))
        h_conv = tf.nn.relu(tf.nn.conv2d(input, w_conv, strides=[1, 1, 1, 1], padding='SAME', name="{}_conv1x1".format(name)) + b_conv,
                            name="{}_conv1x1_relu".format(name))

        return h_conv

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
