"""Cone-beam forward projection

This script is to do cone-beam forward projection given CT data to generate sinograms
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import tensorflow as tf

from pyronn.ct_reconstruction.helpers.misc import generate_sinogram
from pyronn.ct_reconstruction.layers.projection_3d import cone_projection3d
from pyronn.ct_reconstruction.geometry.geometry_cone_3d import GeometryCone3D
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory

from recon_proj_parameters import PROJECTION_PARA, NORMALIZATION_VOLUME_SPACING


def forward_projections():
    """
    main procedures for cone-beam forward projection
    """

    phantoms_dir = 'normalized_ct_phantoms/'
    num_phantoms = len(os.listdir(phantoms_dir))

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True
    # ------------------ Call Layers ------------------
    with tf.Session(config=config):
        for n in range(num_phantoms):
            phantom_file = phantoms_dir + 'phantom_' + str(n+1) + '.npy'
            phantom = np.load(phantom_file)

            # Volume Parameters:
            volume_shape = [phantom.shape[0], phantom.shape[1], phantom.shape[2]]
            volume_spacing = NORMALIZATION_VOLUME_SPACING

            # Detector Parameters:
            detector_shape = PROJECTION_PARA['detector_shape']
            detector_spacing = PROJECTION_PARA['detector_spacing']

            # Trajectory Parameters:
            number_of_projections = PROJECTION_PARA['number_of_projections']
            angular_range = PROJECTION_PARA['angular_range']

            source_detector_distance =  PROJECTION_PARA['source_detector_distance']
            source_isocenter_distance = PROJECTION_PARA['source_isocenter_distance']

            # create Geometry class
            geometry = GeometryCone3D(volume_shape, volume_spacing, detector_shape, detector_spacing, number_of_projections,
                                      angular_range, source_detector_distance, source_isocenter_distance)
            projection_geometry = circular_trajectory.circular_trajectory_3d(geometry)
            geometry.set_projection_matrices(projection_geometry)

            # generate projection sinogram
            sinogram = generate_sinogram.generate_sinogram(phantom, cone_projection3d, geometry)
            np.save('sinograms/sinogram_' + str(n+1), sinogram)
            print('Sinogram ' + str(n+1) + ' generated!')


def show_sinogram(sinogram_index):
    """
    display the generated sinograms

    Parameters
    ----------
    sinogram_index : int
       The index for which CT's sinograms to display
    """

    sinogram_dir = 'sinograms/sinogram_' + str(sinogram_index) + '.npy'
    sinogram = np.load(sinogram_dir)

    fig = plt.figure()

    imgs = []
    for i in range(sinogram.shape[0]):
        img = plt.imshow(sinogram[i,:,:], animated=True, cmap=plt.get_cmap('gist_gray'))
        imgs.append([img])

    animation.ArtistAnimation(fig, imgs, interval=50, blit=True, repeat_delay=1000)

    plt.show()


if __name__ == '__main__':
    ###############################
    # do cone-beam forward projection
    # forward_projections()

    ###############################
    # display the projection images for test
    show_sinogram(20)