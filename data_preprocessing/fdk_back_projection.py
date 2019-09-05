"""FDK backprojection

This script is to do reconstruction from projection data via the FDK algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

import tensorflow as tf
import dicom

from pyronn.ct_reconstruction.layers.backprojection_3d import cone_backprojection3d
from pyronn.ct_reconstruction.geometry.geometry_cone_3d import GeometryCone3D
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory
from pyronn.ct_reconstruction.helpers.filters.filters import ram_lak_3D
import pyronn.ct_reconstruction.helpers.filters.weights as ct_weights

from recon_proj_parameters import PROJECTION_PARA,RECONSTRUCT_PARA


class FDKBackProjection:
    """
    The class for the FDK algorithm for reconstruction
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
                                             initializer=ct_weights.cosine_weights_3d(self.geometry), trainable=False)

        self.filter = tf.get_variable(name='reco_filter', dtype=tf.float32, initializer=ram_lak_3D(self.geometry), trainable=False)

    def model(self, sinogram):
        """
        main model for the FDK algorithm

        Parameters
        ----------
        sinogram : ndarray
            The projection data used for reconstruction

        Returns
        -------
        ndarray
            the reconstructed CT data
        """

        sinogram_cos = tf.multiply(sinogram, self.cosine_weight)

        weighted_sino_fft = tf.fft(tf.cast(sinogram_cos, dtype=tf.complex64))
        filtered_sinogram_fft = tf.multiply(weighted_sino_fft, tf.cast(self.filter,dtype=tf.complex64))
        filtered_sinogram = tf.real(tf.ifft(filtered_sinogram_fft))

        reconstruction = cone_backprojection3d(filtered_sinogram, self.geometry, hardware_interp=True)

        return reconstruction

    def update_geometry_volume_spacing(self, volume_spacing):
        """
        update the geometry value

        Parameters
        ----------
        volume_spacing : list
            The volume spacing values to update for geometry
        """

        self.geometry.volume_spacing = np.array(volume_spacing, dtype=np.float32)
        self.geometry.volume_origin = -(self.geometry.volume_shape - 1) / 2.0 * self.geometry.volume_spacing


def load_volume_size_list():
    """
    calculate and store the volume spacing values for each CT sample

    Returns
    -------
    list
        the list containing all the volume spacing values for each CT by sequence of index
    """

    phantoms_dir = '../3Dircadb1/'
    num_phantoms = len(os.listdir(phantoms_dir))

    volume_size_list = []
    for n in range(num_phantoms):
        phantom_dir = phantoms_dir + '3Dircadb1.' + str(n + 1) + '/PATIENT_DICOM/'
        num_slices = len(os.listdir(phantom_dir))

        dcm = dicom.read_file(phantom_dir + "image_0")
        num_row = dcm.Rows
        num_col = dcm.Columns
        row_pixel_spacing = np.round(dcm.PixelSpacing[0], 2)
        col_pixel_spacing = np.round(dcm.PixelSpacing[1], 2)
        slice_thickness = np.round(dcm.SliceThickness, 2)

        # Volume Parameters:
        volume_size = [num_slices*slice_thickness/RECONSTRUCT_PARA['volume_shape'][0],
                       num_row*row_pixel_spacing/RECONSTRUCT_PARA['volume_shape'][1],
                       num_col*col_pixel_spacing/RECONSTRUCT_PARA['volume_shape'][2]]
        volume_size_list.append(volume_size)

    return volume_size_list


def normalize_sino(sinogram):
    """
    normalize the projection data to [0, 255]

    Parameters
    ----------
    sinogram : ndarray
        The projection data to normalize

    Returns
    -------
    ndarray
        the normalized projection data
    """

    min = np.min(sinogram)
    max = np.max(sinogram)

    sinogram = (sinogram - min) * 255 / (max - min)

    return sinogram


def reconstruction(limited_angle_num):
    """
    procedures for reconstruction from projection data

    Parameters
    ----------
    limited_angle_num : int
        The limited angle for the reconstruction
    """

    volume_size_list = load_volume_size_list()

    # Volume Parameters:
    volume_shape = RECONSTRUCT_PARA['volume_shape']
    volume_spacing = RECONSTRUCT_PARA['volume_spacing']

    # Detector Parameters:
    detector_shape = PROJECTION_PARA['detector_shape']
    detector_spacing = PROJECTION_PARA['detector_spacing']

    # Trajectory Parameters:
    number_of_projections = limited_angle_num + 1
    angular_range = np.radians(limited_angle_num)

    source_detector_distance = PROJECTION_PARA['source_detector_distance']
    source_isocenter_distance = PROJECTION_PARA['source_isocenter_distance']

    # create Geometry class
    geometry = GeometryCone3D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections, angular_range, source_detector_distance, source_isocenter_distance)
    projection_geometry = circular_trajectory.circular_trajectory_3d(geometry)
    geometry.set_projection_matrices(projection_geometry)

    ############################# number of phantoms to reconstruct
    sinograms_dir = 'sinograms/'
    num_phantoms = len(os.listdir(sinograms_dir))

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    # ------------------ Call Layers ------------------
    with tf.Session(config=config) as sess:
        model = FDKBackProjection(geometry)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for n in range(num_phantoms):
            sinogram_dir = sinograms_dir + 'sinogram_' + str(n + 1) + '.npy'
            sinogram = np.load(sinogram_dir)
            sinogram = normalize_sino(sinogram[:number_of_projections,:,:])

            model.update_geometry_volume_spacing(volume_size_list[n])

            recon = model.model(sinogram)
            recon_fdk = sess.run(recon)

            np.save('recon_' + str(limited_angle_num) + '/recon_' + str(n + 1), recon_fdk)
            print('recon_ ' + str(n + 1) + ' generated!')


def show_reconstruction(limited_angle_num, phantom_index):
    """
    display the reconstructed CT given the limtied angle for reconstruction and the index number

    Parameters
    ----------
    limited_angle_num : int
        The limited angle used for reconstruction

    phantom_index : int
        The index number for the reconstructed CT
    """

    recon_dir = 'recon_' + str(limited_angle_num) + '/recon_' + str(phantom_index) + '.npy'
    recon_fdk = np.load(recon_dir)

    fig = plt.figure()

    imgs = []
    for i in range(recon_fdk.shape[0]):
        img = plt.imshow(recon_fdk[i, :, :], animated=True, cmap=plt.get_cmap('gist_gray'))
        imgs.append([img])

    animation.ArtistAnimation(fig, imgs, interval=50, blit=True, repeat_delay=1000)

    plt.show()


def compare_reconstruction(angle_one, angle_two, phantom_index, slice_index):
    """
    display two reconstructed CT images given the limtied angles for these two CT,
    the CT index number and the slice number

    Parameters
    ----------
    angle_one : int
        The limited angle for the first CT

    angle_two : int
        The limited angle for the second CT

    phantom_index : int
        The index number for the reconstructed CT

    slice_index : int
        The index number for slice in the reconstructed CT
    """

    recon_one = 'recon_' + str(angle_one) + '/recon_' + str(phantom_index) + '.npy'
    recon_one = np.load(recon_one)
    recon_one = recon_one[slice_index-1,:,:]

    recon_two = 'recon_' + str(angle_two) + '/recon_' + str(phantom_index) + '.npy'
    recon_two = np.load(recon_two)
    recon_two = recon_two[slice_index-1,:,:]

    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(recon_one, cmap=plt.get_cmap('gist_gray'))
    ax.set_title('Reconstruction for limited angle ' + str(angle_one))

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(recon_two, cmap=plt.get_cmap('gist_gray'))
    ax.set_title('Reconstruction for limited angle ' + str(angle_two))

    plt.show()


if __name__ == '__main__':
    ##########################################################################
    # do reconstruction from a limited angle of 145, 180 and 360 degrees
    # reconstruction(145)
    # reconstruction(180)
    # reconstruction(360)

    ##########################################################################
    # show reconstruction
    show_reconstruction(145, 20) # 145 180 360

    ##########################################################################
    # compare reconstruction
    # compare_reconstruction(145, 360, 20, 70)