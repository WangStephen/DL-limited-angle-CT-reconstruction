"""CT Preprocessing

This script is to preprocess the CT data

first load the CT data and then resample the CT data to the same spacing size
next adjust the Hounsfield Unit scale to the range [-1000, 400]

It also provides the function to display the CT after preprocessing
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import dicom

from scipy import ndimage
from recon_proj_parameters import NORMALIZATION_VOLUME_SPACING


def load_ct_phantom(phantom_dir):
    """
    load the CT data from a directory

    Parameters
    ----------
    phantom_dir : str
        The directory contianing the CT data to load

    Returns
    -------
    ndarray
        the CT data array
    list
        the spacing property for this CT
    """

    # dicom parameters
    dcm = dicom.read_file(phantom_dir + "image_0")
    row_pixel = dcm.Rows
    col_pixel = dcm.Columns
    row_pixel_spacing = np.round(dcm.PixelSpacing[0], 2)
    col_pixel_spacing = np.round(dcm.PixelSpacing[1], 2)
    slice_thickness = np.round(dcm.SliceThickness, 2)

    num_slices = len(os.listdir(phantom_dir))
    phantom = np.zeros((num_slices, row_pixel, col_pixel))
    for i in range(num_slices):
        dcm = dicom.read_file(phantom_dir + "image_" + str(i))
        dcm.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        phantom[i, :, :] = dcm.image.copy()

    # Volume Parameters:
    volume_size = [slice_thickness, row_pixel_spacing, col_pixel_spacing]

    return phantom, volume_size


def resample_phantom(phantom, volume_size):
    """
    resample the CT to a certain normalized spacing sizes

    Parameters
    ----------
    phantom : ndarray
        The CT data to resample

    volume_size : list
        The spacing property for this CT

    Returns
    -------
    ndarray
        the CT after resampling
    """

    z_scale = volume_size[0] / NORMALIZATION_VOLUME_SPACING[0]
    row_scale = volume_size[1] / NORMALIZATION_VOLUME_SPACING[1]
    col_scale = volume_size[2] / NORMALIZATION_VOLUME_SPACING[2]

    resampled_phantom = ndimage.interpolation.zoom(phantom, [z_scale, row_scale, col_scale], mode='nearest')

    return resampled_phantom


def adjust_phantom_HU(phantom):
    """
    adjust the HU scale for the CT data to range [-1000, 400]

    Parameters
    ----------
    phantom : ndarray
        The CT data to adjust

    Returns
    -------
    ndarray
        the CT after adjusting
    """

    min_bound = -1000
    max_bound = 400

    phantom[phantom < min_bound] = min_bound
    phantom[phantom > max_bound] = max_bound

    return phantom


def save_phantom(phantom, index):
    """
    save the CT data after preprocessing

    Parameters
    ----------
    phantom : ndarray
        The CT data to save

    index : int
        The index number for this CT for saved name
    """

    np.save('normalized_ct_phantoms/phantom_' + str(index), phantom)
    print('Normalized phantom ' + str(index) + ' stored!')


def main_preprocessing(phantoms_dir):
    """
    main procedures for the preprocessing

    Parameters
    ----------
    phantoms_dir : str
        The directory for all the CT phantoms
    """

    num_phantoms = len(os.listdir(phantoms_dir))
    for n in range(num_phantoms):
        phantom_dir = phantoms_dir + '3Dircadb1.' + str(n + 1) + '/PATIENT_DICOM/'

        phantom, volume_size = load_ct_phantom(phantom_dir)
        phantom = resample_phantom(phantom, volume_size)
        phantom = adjust_phantom_HU(phantom)

        save_phantom(phantom, n+1)


def show_phantom(phantom_index):
    """
    display the CT after preprocessing

    Parameters
    ----------
    phantom_index : int
        The index number for which CT to display
    """

    phantom_dir = 'normalized_ct_phantoms/phantom_' + str(phantom_index) + '.npy'
    phantom = np.load(phantom_dir)

    fig = plt.figure()

    imgs = []
    for i in range(phantom.shape[0]):
        img = plt.imshow(phantom[i, :, :], animated=True, cmap=plt.get_cmap('gist_gray'))
        imgs.append([img])

    animation.ArtistAnimation(fig, imgs, interval=50, blit=True, repeat_delay=1000)

    plt.show()


if __name__ == '__main__':
    ##############################################################
    # preprocessing main procedures
    # phantoms_dir = '../3Dircadb1/'
    # main_preprocessing(phantoms_dir)

    ##############################################################
    # display normalized phantoms after preprocessing
    show_phantom(20)