"""loading training, validation and test data

This script is to load training, validation and test data
"""

import numpy as np
import os
import dicom

from geometry_parameters import GEOMETRY, RECONSTRUCT_PARA
from geometry_parameters import TRAIN_INDEX, VALID_INDEX, TEST_INDEX
from geometry_parameters import NUM_TRAINING_SAMPLES, NUM_VALIDATION_SAMPLES, NUM_TEST_SAMPLES


def load_training_data():
    """
    load training data

    Returns
    -------
    ndarray
        training data
    ndarray
        training ground truth
    """

    train_data_numpy = np.empty((NUM_TRAINING_SAMPLES,) + tuple(GEOMETRY.sinogram_shape))
    train_labels_numpy = np.empty((NUM_TRAINING_SAMPLES,) + tuple(GEOMETRY.volume_shape))
    i = 0
    for index in TRAIN_INDEX:
        train_data_file = '../data_preprocessing/sinograms/sinogram_' + str(index) + '.npy'
        train_data_numpy[i, :, :, :] = np.load(train_data_file)[:GEOMETRY.number_of_projections, :, :]
        train_label_file = '../data_preprocessing/recon_360/recon_' + str(index) + '.npy'
        train_labels_numpy[i, :, :, :] = np.load(train_label_file)
        i = i + 1

    return train_data_numpy, train_labels_numpy


def load_validation_data():
    """
    load validation data

    Returns
    -------
    ndarray
        validation data
    ndarray
        validation ground truth
    """

    validation_data_numpy = np.empty((NUM_VALIDATION_SAMPLES,) + tuple(GEOMETRY.sinogram_shape))
    validation_labels_numpy = np.empty((NUM_VALIDATION_SAMPLES,) + tuple(GEOMETRY.volume_shape))
    i = 0
    for index in VALID_INDEX:
        valid_data_file = '../data_preprocessing/sinograms/sinogram_' + str(index) + '.npy'
        validation_data_numpy[i, :, :, :] = np.load(valid_data_file)[:GEOMETRY.number_of_projections, :, :]
        valid_label_file = '../data_preprocessing/recon_360/recon_' + str(index) + '.npy'
        validation_labels_numpy[i, :, :, :] = np.load(valid_label_file)
        i = i + 1

    return validation_data_numpy, validation_labels_numpy


def load_test_data():
    """
    load test data

    Returns
    -------
    ndarray
        test data
    ndarray
        test ground truth
    """

    test_data_numpy = np.empty((NUM_TEST_SAMPLES,) + tuple(GEOMETRY.sinogram_shape))
    test_labels_numpy = np.empty((NUM_TEST_SAMPLES,) + tuple(GEOMETRY.volume_shape))
    i = 0
    for index in TEST_INDEX:
        test_data_file = '../data_preprocessing/sinograms/sinogram_' + str(index) + '.npy'
        test_data_numpy[i, :, :, :] = np.load(test_data_file)[:GEOMETRY.number_of_projections, :, :]
        test_label_file = '../data_preprocessing/recon_360/recon_' + str(index) + '.npy'
        test_labels_numpy[i, :, :, :] = np.load(test_label_file)
        i = i + 1

    return test_data_numpy, test_labels_numpy


def load_voxel_size_list():
    """
    load voxel sizes for each CT

    Returns
    -------
    list
        a list containing the voxel sizes needed for reconstruction for models training
    """

    phantoms_dir = '../3Dircadb1/'
    num_phantoms = len(os.listdir(phantoms_dir))

    voxel_size_list = []
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
        voxel_size = [num_slices * slice_thickness / RECONSTRUCT_PARA['volume_shape'][0],
                      num_row * row_pixel_spacing / RECONSTRUCT_PARA['volume_shape'][1],
                      num_col * col_pixel_spacing / RECONSTRUCT_PARA['volume_shape'][2]]
        voxel_size_list.append(voxel_size)

    return voxel_size_list