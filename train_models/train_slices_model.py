"""training model with separate U-Net

This script is to train a model with separate U-Net
"""

import numpy as np
import datetime
import tensorflow as tf

from pyronn.ct_reconstruction.geometry.geometry_cone_3d import GeometryCone3D
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory

from geometry_parameters import GEOMETRY, TRAIN_INDEX, VALID_INDEX, TEST_INDEX
from geometry_parameters import NUM_TRAINING_SAMPLES, NUM_VALIDATION_SAMPLES, NUM_TEST_SAMPLES
from geometry_parameters import RECONSTRUCT_PARA, PROJECTION_PARA

import load_data
####################################### Create a list of Geometry
voxel_size_list = load_data.load_voxel_size_list()

GEO_LIST = []
for n in range(len(voxel_size_list)):
    GEO_LIST.append(GeometryCone3D(RECONSTRUCT_PARA['volume_shape'], voxel_size_list[n],
                  PROJECTION_PARA['detector_shape'], PROJECTION_PARA['detector_spacing'],
                  PROJECTION_PARA['number_of_projections'], PROJECTION_PARA['angular_range'],
                  PROJECTION_PARA['source_detector_distance'], PROJECTION_PARA['source_isocenter_distance']))
    GEO_LIST[n].set_projection_matrices(circular_trajectory.circular_trajectory_3d(GEO_LIST[n]))
##################################################################

from unet_reconstruction_model import model_unet_slices


# training parameters
LEARNING_RATE = 1e-5
MAX_EPOCHS = 800

# set graph seed
tf.random.set_random_seed(13)


class Pipeline:

    def __init__(self, model_name):
        """
        Parameters
        ----------
        model_name : str
           The model to train
        """

        self.model_name = model_name
        self.model = model_unet_slices.ModelProposedNet(GEOMETRY)

    def build_inital_graph(self):
        """
        build the first initialization training graph
        """

        # Placeholders for data and label
        self.data_placeholder = tf.placeholder(tf.float32, shape=(None,) + tuple(GEOMETRY.sinogram_shape))
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(None,) + tuple(GEOMETRY.volume_shape))
        self.index_placeholder = tf.placeholder(tf.int64)

        # Create tf dataset from placholders
        dataset = tf.data.Dataset.from_tensor_slices((self.data_placeholder, self.labels_placeholder, self.index_placeholder))

        # Create a initializable dataset iterator
        self.iter = dataset.make_initializable_iterator()
        self.sinograms, self.labels, self.index = self.iter.get_next()

        # Create a Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        # Saver
        self.saver = tf.train.Saver(max_to_keep=1)

    def build_model_proj_graph(self):
        """
        build the training graph for the projection domain
        """

        self.filtered_sino = self.model.forward_proj_domain(self.sinograms)

    def build_model_recon_graph(self):
        """
        build the training graph for the reconstruction domain
        """

        self.reconstruction = self.model.forward_recon_domain(self.filtered_sino, self.index)
        self.loss = self.calculate_loss(self.reconstruction, self.labels)

    def build_train_op_graph(self):
        """
        build the training graph for optimization
        """

        # run optimization
        self.train_op = self.optimizer.minimize(self.loss)

    def calculate_loss(self, predictions, gt_labels):
        """
        calculate mse loss

        Parameters
        ----------
        predictions : ndarray
            The predicted results

        gt_labels : ndarray
            The ground truth

        Returns
        -------
        float
            the mse loss
        """

        loss = tf.reduce_mean(tf.squared_difference(predictions, gt_labels))

        return loss

    def do_model_eval(self, sess, input, labels, steps, phantom_index_list, save_recon):
        """
        do evaluation on model

        Parameters
        ----------
        sess : session
            evaluation on which session

        input : ndarray
            input for evaluation

        labels : ndarray
            The ground truth

        steps : ndarray
            number of data for evaluation

        phantom_index_list : ndarray
            index for the input data

        save_recon : ndarray
            list of whether to save the reconstructed evaluation results and which folder to save them

        Returns
        -------
        float
            the evaluation loss
        """

        # Initialize dataset iterator with data
        sess.run(self.iter.initializer, feed_dict={self.data_placeholder: input,
                                                   self.labels_placeholder: labels,
                                                   self.index_placeholder: phantom_index_list})

        # Run model and calculate avg loss for set
        losses = 0
        for i in range(steps):
            # run the model
            index, recon, loss = sess.run([self.index, self.reconstruction, self.loss])

            # save the evaluation recon
            if save_recon[0]:
                np.save(save_recon[1] + 'recon_' + str(index), recon)
                print('recon_' + str(index) + ' generated!')

            # sum the loss
            losses += np.mean(loss)

        return losses / steps

    def normalize_sino(self, sinogram):
        """
        normalize all the sinograms to [0, 255]

        Parameters
        ----------
        sinogram : ndarray
            The input sinograms to normalize

        Returns
        -------
        ndarray
            the normalized sinograms results
        """

        for i in range(sinogram.shape[0]):
            min = np.min(sinogram[i,::])
            max = np.max(sinogram[i,::])

            sinogram[i,::] = (sinogram[i,::] - min) * 255 / (max - min)

        return sinogram

    def normalize_labels(self, labels):
        """
        normalize all the ground truth to [0, 1]

        Parameters
        ----------
        labels : ndarray
            The input ground truth to normalize

        Returns
        -------
        ndarray
            the normalized ground truth results
        """

        for i in range(labels.shape[0]):
            min = np.min(labels[i,::])
            max = np.max(labels[i,::])

            labels[i,::] = (labels[i,::] - min) / (max - min)

        return labels

    def generate_recons(self):
        """
        generate all reconstructed CT samples from the FDK neural network which will be used for later training in U-Net
        """

        # load all the data
        train_data_numpy, train_labels_numpy = load_data.load_training_data()
        validation_data_numpy, validation_labels_numpy = load_data.load_validation_data()
        test_data_numpy, test_labels_numpy = load_data.load_test_data()

        # normalize the input data
        train_data_numpy = self.normalize_sino(train_data_numpy)
        validation_data_numpy = self.normalize_sino(validation_data_numpy)
        test_data_numpy = self.normalize_sino(test_data_numpy)

        # normalize the labels
        train_labels_numpy = self.normalize_labels(train_labels_numpy)
        validation_labels_numpy = self.normalize_labels(validation_labels_numpy)
        test_labels_numpy = self.normalize_labels(test_labels_numpy)

        # session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # Build Graph
            self.build_inital_graph()
            self.build_model_proj_graph()
            self.build_model_recon_graph()
            self.build_train_op_graph()

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # generation on set
            print('\n############################### generating')
            best_model_sess_file = tf.train.latest_checkpoint('fdk_nn_model/saved_session/')
            self.saver.restore(sess, best_model_sess_file)

            self.do_model_eval(sess, train_data_numpy, train_labels_numpy,
                               NUM_TRAINING_SAMPLES, TRAIN_INDEX, [True, self.model_name + '/eval_recon/generation_recons/'])
            self.do_model_eval(sess, validation_data_numpy, validation_labels_numpy,
                               NUM_VALIDATION_SAMPLES, VALID_INDEX, [True, self.model_name + '/eval_recon/generation_recons/'])
            self.do_model_eval(sess, test_data_numpy, test_labels_numpy,
                                NUM_TEST_SAMPLES, TEST_INDEX, [True, self.model_name  + '/eval_recon/generation_recons/'])

    def load_recons(self, num, data_index):
        """
        load the generated reconstructed CT samples

        Parameters
        ----------
        num : int
            The number of CT to load

        data_index : list
            The index of CT to load

        Returns
        -------
        ndarray
            the CT data

        ndarray
            the ground truth for each CT
        """

        data_numpy = np.empty((num,) + tuple(GEOMETRY.volume_shape))
        labels_numpy = np.empty((num,) + tuple(GEOMETRY.volume_shape))
        i = 0
        for index in data_index:
            data_file = 'unet_reconstruction_model/eval_recon/generation_recons/recon_' + str(index) + '.npy'
            data_numpy[i, :, :, :] = np.load(data_file)
            label_file = '../data_preprocessing/recon_360/recon_' + str(index) + '.npy'
            labels_numpy[i, :, :, :] = np.load(label_file)
            i = i + 1

        return data_numpy, labels_numpy

    def into_slices(self, input):
        """
        split all the CT samples into one sample with all the slices

        Parameters
        ----------
        input : ndarray
            The CT samples to split

        Returns
        -------
        ndarray
            all the slices from the input CT samples
        """

        num_slices = input.shape[0] * input.shape[1]
        output = np.empty((num_slices, GEOMETRY.volume_shape[1], GEOMETRY.volume_shape[2]))

        i = 0
        for k in range(input.shape[0]):
            for m in range(input.shape[1]):
                output[i,::] = input[k,m,::]
                i = i + 1

        return output

    def build_training_slices_graph(self):
        """
        build the training graph for slice-to-slice training
        """

        # Placeholders for recons and gts
        self.recons_placeholder = tf.placeholder(tf.float32, shape=(None, GEOMETRY.volume_shape[1], GEOMETRY.volume_shape[2]))
        self.gts_placeholder = tf.placeholder(tf.float32, shape=(None, GEOMETRY.volume_shape[1], GEOMETRY.volume_shape[2]))

        # Create tf dataset from placholders
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.recons_placeholder, self.gts_placeholder)).batch(self.batch_size)

        # Create a initializable dataset iterator
        self.iterator = dataset.make_initializable_iterator()
        self.recons, self.gts = self.iterator.get_next()

        # Create a Optimizer
        self.optimizer_adam = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        # Saver
        self.saver_sess_slice = tf.train.Saver(max_to_keep=1)

        # graph to train
        self.unet_recons = self.model.unet_model(self.recons)
        self.losses = self.calculate_loss(self.unet_recons, self.gts)

        # run optimization
        self.train_ops = self.optimizer_adam.minimize(self.losses)

    def do_slices_eval(self, sess, input, labels, steps, save_recon):
        """
        do evaluation on model for slice-to-slice training

        Parameters
        ----------
        sess : session
            evaluation on which session

        input : ndarray
            input for evaluation

        labels : ndarray
            The ground truth

        steps : ndarray
            number of data for evaluation

        save_recon : ndarray
            list of whether to save the reconstructed evaluation results and which folder to save them

        Returns
        -------
        float
            the evaluation loss
        """

        # Initialize dataset iterator with data
        sess.run(self.iterator.initializer, feed_dict={self.recons_placeholder: input,
                                                   self.gts_placeholder: labels})

        # Run model and calculate avg loss for set
        losses = 0
        for i in range(int(steps)):
            # run the model
            recon, loss = sess.run([self.unet_recons, self.losses])

            # save the evaluation recon
            if save_recon[0]:
                np.save(save_recon[1] + 'recon_' + str(i), recon)
                print('recon_' + str(i) + ' generated!')

            # sum the loss
            losses += np.mean(loss)

        return losses / steps

    def run_training(self):
        """
        do slice-to-slice training using unet.
        """

        # load all the reconstructed CT data from the FDK neural network
        train_data_numpy, train_labels_numpy = self.load_recons(NUM_TRAINING_SAMPLES, TRAIN_INDEX)
        validation_data_numpy, validation_labels_numpy = self.load_recons(NUM_VALIDATION_SAMPLES, VALID_INDEX)
        test_data_numpy, test_labels_numpy = self.load_recons(NUM_TEST_SAMPLES, TEST_INDEX)

        # normalize the labels
        train_labels_numpy = self.normalize_labels(train_labels_numpy)
        validation_labels_numpy = self.normalize_labels(validation_labels_numpy)
        test_labels_numpy = self.normalize_labels(test_labels_numpy)

        # into slices
        train_data_numpy = self.into_slices(train_data_numpy)
        train_labels_numpy = self.into_slices(train_labels_numpy)
        validation_data_numpy = self.into_slices(validation_data_numpy)
        validation_labels_numpy = self.into_slices(validation_labels_numpy)
        test_data_numpy = self.into_slices(test_data_numpy)
        test_labels_numpy = self.into_slices(test_labels_numpy)

        self.batch_size = 10
        # Training session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # Build Graph
            self.build_training_slices_graph()

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            valid_losses = []
            best_valid_loss = np.Inf
            for epoch in range(MAX_EPOCHS):
                # Initialise dataset iterator
                sess.run(self.iterator.initializer, feed_dict={self.recons_placeholder: train_data_numpy,
                                                           self.gts_placeholder: train_labels_numpy})

                training_losses = 0
                for step in range(int(train_data_numpy.shape[0]/self.batch_size)):
                    # run the training
                    training_loss, _ = sess.run([self.losses, self.train_ops])
                    training_losses += np.mean(training_loss)

                valid_loss = self.do_slices_eval(sess, validation_data_numpy, validation_labels_numpy,
                                                validation_data_numpy.shape[0]/self.batch_size, [False, ''])
                valid_losses.append(valid_loss)

                print('Epoch: {0:3d}/{1:3d}, training loss: {2:3.8f}, validation loss: {3:3.8f}'
                      .format(epoch + 1, MAX_EPOCHS, training_losses / (train_data_numpy.shape[0]/self.batch_size), valid_loss))

                # early stopping if validation loss is increasing or staying the same after five epoches
                last_five_valid_losses = valid_losses[-5:]
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss

                    # Save a checkpoint of the least validation loss model so far
                    # print("saving this least validation loss model so far!")
                    self.saver_sess_slice.save(sess, self.model_name + '/saved_session/sess-' +
                                    '{date:%m_%d_%H:%M}'.format(date=datetime.datetime.now()) +
                                    '.ckpt', global_step=epoch)
                elif len(last_five_valid_losses) == 5 and all([valid_loss >= x for x in last_five_valid_losses]):
                    # print('early stopping !!!')
                    break
                else:
                    # print('no improvement on validation at this epoch, continue training...')
                    continue

            # evaluate on test set
            print('\n############################### testing evaluation on best trained model so far')
            best_model_sess_file = tf.train.latest_checkpoint(self.model_name + '/saved_session/')
            self.saver_sess_slice.restore(sess, best_model_sess_file)

            test_loss = self.do_slices_eval(sess, test_data_numpy, test_labels_numpy,
                                           np.ceil(test_data_numpy.shape[0]/self.batch_size), [True, self.model_name + '/eval_recon/'])
            print("average test loss: ", test_loss)


if __name__ == "__main__":
    the_pipeline = Pipeline('unet_reconstruction_model')
    # the_pipeline.generate_recons()
    # the_pipeline.run_training()

