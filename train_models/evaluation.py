"""Evaluation

This script consists of evaluation functions needed
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime

import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp

import load_data
from geometry_parameters import TEST_INDEX, RECONSTRUCT_PARA


def show_reconstruction(model, phantom_index):
    """
    show reconstructed CT

    Parameters
    ----------
    model : str
        which model's results to use

    phantom_index : int
        which CT to display
    """

    recon_dir = model + '/eval_recon/recon_' + str(phantom_index) + '.npy'
    recon = np.load(recon_dir)

    fig = plt.figure()

    imgs = []
    for i in range(recon.shape[0]):
        img = plt.imshow(recon[i, :, :], animated=True, cmap=plt.get_cmap('gist_gray'))
        imgs.append([img])

    animation.ArtistAnimation(fig, imgs, interval=50, blit=True, repeat_delay=1000)

    plt.show()


def compare_reconstruction(model_one, model_two, phantom_index, slice_index):
    """
    compared reconstructed CT results from different two models

    Parameters
    ----------
    model_one : str
        the first model's result to use

    model_two : str
        the second model's result to use

    phantom_index : int
        which CT to display

    slice_index : int
        which slice in the CT to display
    """

    recon_one = model_one + '/eval_recon/recon_' + str(phantom_index) + '.npy'
    recon_one = np.load(recon_one)
    recon_one = recon_one[slice_index-1,:,:]

    recon_two = model_two + '/eval_recon/recon_' + str(phantom_index) + '.npy'
    recon_two = np.load(recon_two)
    recon_two = recon_two[slice_index-1,:,:]

    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(recon_one, cmap=plt.get_cmap('gist_gray'))
    ax.set_title('model: ' + model_one)

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(recon_two, cmap=plt.get_cmap('gist_gray'))
    ax.set_title('model: ' + model_two)

    plt.show()


def single_ct_normalize(input):
    """
    normalize one CT sample to [0, 1]

    Parameters
    ----------
    input : ndarray
        The input CT to normalize

    Returns
    -------
    ndarray
        the normalized CT
    """


    max = np.max(input)
    min = np.min(input)

    input = (input - min) / (max - min)

    return input


def compare_reconstruction_with_fdk(model, phantom_index, slice_index):
    """
    compare reconstructed CT results with the conventional FDK and the ground truth

    Parameters
    ----------
    model : str
        which model's results to use

    phantom_index : int
        which CT to display

    slice_index : int
        which slice in the CT to display
    """

    recon_one = '../data_preprocessing/recon_145/recon_' + str(phantom_index) + '.npy'
    recon_one = single_ct_normalize(np.load(recon_one))
    recon_one = recon_one[slice_index - 1, :, :]

    recon_two = model + '/eval_recon/recon_' + str(phantom_index) + '.npy'
    recon_two = np.load(recon_two)
    recon_two = recon_two[slice_index - 1, :, :]

    recon_three = '../data_preprocessing/recon_360/recon_' + str(phantom_index) + '.npy'
    recon_three = single_ct_normalize(np.load(recon_three))
    recon_three = recon_three[slice_index - 1, :, :]

    fig = plt.figure(figsize=plt.figaspect(0.3))

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(recon_one, cmap=plt.get_cmap('gist_gray'))
    ax.set_title('pure_fdk')

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(recon_two, cmap=plt.get_cmap('gist_gray'))
    ax.set_title('model: ' + model)

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(recon_three, cmap=plt.get_cmap('gist_gray'))
    ax.set_title('ground truth')

    plt.show()


def calculate_ssim(predictions, gt_labels, max_val):
    """
    ssim calculation

    Parameters
    ----------
    predictions : ndarray
        the reconstructed results

    gt_labels : ndarray
        the ground truth

    max_val : float
        the value range
    """

    tf_predictions = tf.placeholder(tf.float32, shape=predictions.shape)
    tf_gt_labels = tf.placeholder(tf.float32, shape=gt_labels.shape)

    tf_ssim_value = tf.image.ssim(tf.expand_dims(tf_predictions, 4),
                                tf.expand_dims(tf_gt_labels, 4), max_val)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        ssim = sess.run(tf_ssim_value, feed_dict={tf_predictions: predictions,
                                                tf_gt_labels: gt_labels})

    return np.mean(ssim)


def calculate_ms_ssim(predictions, gt_labels, max_val):
    """
    ms-ssim calculation

    Parameters
    ----------
    predictions : ndarray
        the reconstructed results

    gt_labels : ndarray
        the ground truth

    max_val : float
        the value range
    """

    tf_predictions = tf.placeholder(tf.float32, shape=predictions.shape)
    tf_gt_labels = tf.placeholder(tf.float32, shape=gt_labels.shape)

    tf_ms_ssim_value = tf.image.ssim_multiscale(tf.expand_dims(tf_predictions, 4),
                                             tf.expand_dims(tf_gt_labels, 4), max_val)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        ms_ssim = sess.run(tf_ms_ssim_value, feed_dict={tf_predictions: predictions,
                                                        tf_gt_labels: gt_labels})

    return np.mean(ms_ssim)


def calculate_psnr(predictions, gt_labels, max_val):
    """
    psnr calculation

    Parameters
    ----------
    predictions : ndarray
        the reconstructed results

    gt_labels : ndarray
        the ground truth

    max_val : float
        the value range
    """

    tf_predictions = tf.placeholder(tf.float32, shape=predictions.shape)
    tf_gt_labels = tf.placeholder(tf.float32, shape=gt_labels.shape)

    tf_psnr_value = tf.image.psnr(tf.expand_dims(tf_predictions, 4),
                                tf.expand_dims(tf_gt_labels, 4), max_val)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        psnr = sess.run(tf_psnr_value, feed_dict={tf_predictions: predictions,
                                                  tf_gt_labels: gt_labels})

    return np.mean(psnr)


def normalize(input):
    """
    normalize more than one CT sample to [0, 1]

    Parameters
    ----------
    input : ndarray
        The input CT samples to normalize

    Returns
    -------
    ndarray
        the normalized CT results
    """

    for i in range(input.shape[0]):
        min_bound = np.min(input[i,::])
        max_bound = np.max(input[i,::])

        input[i,::] = (input[i,::] - min_bound) / (max_bound - min_bound)

    return input


# ms-ssim, psnr, mse
def evaluate_on_metrics(model):
    """
    do evaluation on mse, ssim, ms-ssim and psnr

    Parameters
    ----------
    model : str
        The model for evaluation
    """

    # get the labels
    _, labels = load_data.load_test_data()
    labels = normalize(labels)

    # load the recons on the model
    recon_phantoms = np.empty(labels.shape)
    for i in range(recon_phantoms.shape[0]):
        recon_file = model + '/eval_recon/recon_' + str(TEST_INDEX[i]) + '.npy'
        recon_phantoms[i,:,:,:] = np.load(recon_file)

    # MSE
    mse = np.mean(np.square(recon_phantoms - labels))

    #
    max_val = 1.0

    # SSIM
    ssim = calculate_ssim(recon_phantoms, labels, max_val)

    # MS-SSIM
    ms_ssim = calculate_ms_ssim(recon_phantoms, labels, max_val)

    # Peak Signal-to-Noise Ratio
    psnr = calculate_psnr(recon_phantoms, labels, max_val)

    # print the results
    print('mse value: ', str(mse))
    print('ssim value: ', str(ssim))
    print('ms-ssim value: ', str(ms_ssim))
    print('psnr value: ', str(psnr))

    # save the metrics results
    f = open(model + '/eval_result/metrics_result.txt', 'a+')
    f.write("Model: {0}, Date: {1:%Y-%m-%d_%H:%M:%S} \nMSE: {2:3.8f} \nSSIM: {3:3.8f} \nMS-SSIM: {4:3.8f} \nPSNR: {5:3.8f}\n\n".format(
                        model, datetime.datetime.now(), mse, ssim, ms_ssim, psnr))
    f.close()


def check_stored_sess_var(sess_file, var_name):
    """
    display variable results for trained models in the stored session

    Parameters
    ----------
    sess_file : str
        the stored session file

    var_name : str
        the variable to see
    """

    if var_name == '':
        # print all tensors in checkpoint file (.ckpt)
        chkp.print_tensors_in_checkpoint_file(sess_file, tensor_name='', all_tensors=True)
    else:
        chkp.print_tensors_in_checkpoint_file(sess_file, tensor_name=var_name, all_tensors=False)


def eval_pure_fdk():
    """
    do evaluation on mse, ssim, ms-ssim and psnr for the conventional FDK algorithm
    """

    # get the labels
    _, labels = load_data.load_test_data()
    labels = normalize(labels)

    # load the recons
    recon_phantoms = np.empty(labels.shape)
    for i in range(recon_phantoms.shape[0]):
        recon_file = '../data_preprocessing/recon_145/recon_' + str(TEST_INDEX[i]) + '.npy'
        recon_phantoms[i, :, :, :] = np.load(recon_file)
    recon_phantoms = normalize(recon_phantoms)

    # MSE
    mse = np.mean(np.square(recon_phantoms - labels))

    #
    max_val = 1.0

    # SSIM
    ssim = calculate_ssim(recon_phantoms, labels, max_val)

    # MS-SSIM
    ms_ssim = calculate_ms_ssim(recon_phantoms, labels, max_val)

    # Peak Signal-to-Noise Ratio
    psnr = calculate_psnr(recon_phantoms, labels, max_val)

    # print the results
    print('mse value: ', str(mse))
    print('ssim value: ', str(ssim))
    print('ms-ssim value: ', str(ms_ssim))
    print('psnr value: ', str(psnr))

    # save the metrics results
    f = open('pure_fdk_model/eval_result/metrics_result.txt', 'a+')
    f.write(
        "Model: {0}, Date: {1:%Y-%m-%d_%H:%M:%S} \nMSE: {2:3.8f} \nSSIM: {3:3.8f} \nMS-SSIM: {4:3.8f} \nPSNR: {5:3.8f}\n\n".format(
            'pure_fdk_model', datetime.datetime.now(), mse, ssim, ms_ssim, psnr))
    f.close()


def convert_to_raw_bin(model):
    """
    convert the reconstructed results of the model to raw data file

    Parameters
    ----------
    model : str
        The model for which results to convert
    """

    dir =  model + '/eval_recon/'

    for i in range(len(TEST_INDEX)):
        recon_file = dir + 'recon_' + str(TEST_INDEX[i]) + '.npy'
        recon = np.load(recon_file)

        recon.astype('float32').tofile(dir + 'recon_' + str(TEST_INDEX[i]) + '_float32_' +
                                       str(RECONSTRUCT_PARA['volume_shape'][1]) + 'x' +
                                       str(RECONSTRUCT_PARA['volume_shape'][2]) + 'x' +
                                       str(RECONSTRUCT_PARA['volume_shape'][0]) + '_bin')


if __name__ == "__main__":
    ###########################################
    # show reconstructed result CT
    show_reconstruction('fdk_nn_model', TEST_INDEX[1])
    # show_reconstruction('cnn_projection_model', TEST_INDEX[1])
    # show_reconstruction('cnn_reconstruction_model', TEST_INDEX[1])
    # show_reconstruction('dense_cnn_reconstruction_model', TEST_INDEX[1])
    # show_reconstruction('unet_projection_model', TEST_INDEX[1])
    # show_reconstruction('unet_reconstruction_model', TEST_INDEX[1])
    # show_reconstruction('unet_proposed_reconstruction_model', TEST_INDEX[1])
    # show_reconstruction('combined_projection_reconstruction_model', TEST_INDEX[1])

    ###########################################
    # Evaluation on each model
    # evaluate_on_metrics('fdk_nn_model')
    # evaluate_on_metrics('cnn_projection_model')
    # evaluate_on_metrics('cnn_reconstruction_model')
    # evaluate_on_metrics('dense_cnn_reconstruction_model')
    # evaluate_on_metrics('unet_projection_model')
    # evaluate_on_metrics('unet_reconstruction_model')
    # evaluate_on_metrics('unet_proposed_reconstruction_model')
    # evaluate_on_metrics('combined_projection_reconstruction_model')
    # eval_pure_fdk()

    ###########################################
    # compare_reconstruction results
    # compare_reconstruction('cnn_projection_model', 'unet_projection_model', TEST_INDEX[1], 75)
    # compare_reconstruction_with_fdk('combined_projection_reconstruction_model', TEST_INDEX[1], 75)

    ###########################################
    # generate raw binary reconstruction files
    # convert_to_raw_bin('combined_projection_reconstruction_model')