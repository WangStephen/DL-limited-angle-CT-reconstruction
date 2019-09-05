"""training model

This script is to train a model
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

from fdk_nn_model import model_fdk_nn
from cnn_projection_model import model_cnn_projection
from cnn_reconstruction_model import model_cnn_reconstruction
from dense_cnn_reconstruction_model import model_dense_cnn_reconstruction
from unet_projection_model import model_unet_projection
from unet_reconstruction_model import model_unet_reconstruction
from unet_proposed_reconstruction_model import model_unet_proposed_reconstruction
from combined_projection_reconstruction_model import model_combined_proj_recon


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

        if self.model_name == 'fdk_nn_model':
            self.model = model_fdk_nn.ModelFDKNet(GEOMETRY)
        elif self.model_name == 'cnn_projection_model':
            self.model = model_cnn_projection.ModelProposedNet(GEOMETRY)
        elif self.model_name == 'cnn_reconstruction_model':
            self.model = model_cnn_reconstruction.ModelProposedNet(GEOMETRY)
        elif self.model_name == 'dense_cnn_reconstruction_model':
            self.model = model_dense_cnn_reconstruction.ModelProposedNet(GEOMETRY)
        elif self.model_name == 'unet_projection_model':
            self.model = model_unet_projection.ModelProposedNet(GEOMETRY)
        elif self.model_name == 'unet_reconstruction_model':
            self.model = model_unet_reconstruction.ModelProposedNet(GEOMETRY)
        elif self.model_name == 'unet_proposed_reconstruction_model':
            self.model = model_unet_proposed_reconstruction.ModelProposedNet(GEOMETRY)
        elif self.model_name == 'combined_projection_reconstruction_model':
            self.model = model_combined_proj_recon.ModelProposedNet(GEOMETRY)

    def build_inital_graph(self):
        """
        build the first initialization training graph
        """

        # Placeholders for data and label and CT index
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

    def run_training(self):
        """
        do training
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

        # Training session
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

            valid_losses = []
            best_valid_loss = np.Inf
            for epoch in range(MAX_EPOCHS):
                # Initialise dataset iterator
                sess.run(self.iter.initializer, feed_dict={self.data_placeholder: train_data_numpy,
                                                           self.labels_placeholder: train_labels_numpy,
                                                           self.index_placeholder: TRAIN_INDEX})

                training_losses = 0
                for step in range(NUM_TRAINING_SAMPLES):
                    # run the training
                    training_loss, _ = sess.run([self.loss, self.train_op])
                    training_losses += np.mean(training_loss)

                valid_loss = self.do_model_eval(sess, validation_data_numpy, validation_labels_numpy,
                                                NUM_VALIDATION_SAMPLES, VALID_INDEX, [False, ''])
                valid_losses.append(valid_loss)

                print('Epoch: {0:3d}/{1:3d}, training loss: {2:3.8f}, validation loss: {3:3.8f}'
                      .format(epoch+1, MAX_EPOCHS, training_losses/NUM_TRAINING_SAMPLES, valid_loss))

                # early stopping if validation loss is increasing or staying the same after five epoches
                last_five_valid_losses = valid_losses[-5:]
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss

                    # Save a checkpoint of the least validation loss model so far
                    # print("saving this least validation loss model so far!")
                    self.saver.save(sess, self.model_name + '/saved_session/sess-' +
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
            best_model_sess_file = tf.train.latest_checkpoint(self.model_name  + '/saved_session/')
            self.saver.restore(sess, best_model_sess_file)

            test_loss = self.do_model_eval(sess, test_data_numpy, test_labels_numpy,
                                            NUM_TEST_SAMPLES, TEST_INDEX, [True, self.model_name  + '/eval_recon/'])
            print("average test loss: ", test_loss)


if __name__ == "__main__":
    # select which model to train
    # the_pipeline = Pipeline('fdk_nn_model')
    # the_pipeline = Pipeline('cnn_projection_model')
    # the_pipeline = Pipeline('cnn_reconstruction_model')
    # the_pipeline = Pipeline('dense_cnn_reconstruction_model')
    # the_pipeline = Pipeline('unet_projection_model')
    the_pipeline = Pipeline('unet_reconstruction_model')
    # the_pipeline = Pipeline('unet_proposed_reconstruction_model')
    # the_pipeline = Pipeline('combined_projection_reconstruction_model')

    the_pipeline.run_training()

