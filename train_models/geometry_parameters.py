import numpy as np

from pyronn.ct_reconstruction.geometry.geometry_cone_3d import GeometryCone3D
from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory


####################################### Projection parameters
# Declare Parameters
PROJECTION_PARA = {}

# Detector Parameters:
PROJECTION_PARA['detector_shape'] = [256, 256]
PROJECTION_PARA['detector_spacing'] = [2.6,2.6]

# Trajectory Parameters:
PROJECTION_PARA['number_of_projections'] = 146
PROJECTION_PARA['angular_range'] = np.radians(145)

PROJECTION_PARA['source_detector_distance'] = 1200
PROJECTION_PARA['source_isocenter_distance'] = 1200


####################################### Reconstruction parameters
# Declare Parameters
RECONSTRUCT_PARA = {}

# Volume Parameters:
RECONSTRUCT_PARA['volume_shape'] = [150, 256, 256]
RECONSTRUCT_PARA['volume_spacing'] = [1.6, 0.73, 0.73]


####################################### Normalization spacing parameters
NORMALIZATION_VOLUME_SPACING = [1.0, 0.56, 0.56]


####################################### Training, validation and test data phantom index
TRAIN_INDEX = [1,2,3,5,6,7,8,9,11,12,13,14,15,16,19]
VALID_INDEX = [10,17,18]
TEST_INDEX = [4,20]

NUM_TRAINING_SAMPLES = len(TRAIN_INDEX)
NUM_VALIDATION_SAMPLES = len(VALID_INDEX)
NUM_TEST_SAMPLES = len(TEST_INDEX)


####################################### Create Geometry
GEOMETRY = GeometryCone3D(RECONSTRUCT_PARA['volume_shape'], RECONSTRUCT_PARA['volume_spacing'],
                          PROJECTION_PARA['detector_shape'], PROJECTION_PARA['detector_spacing'],
                          PROJECTION_PARA['number_of_projections'], PROJECTION_PARA['angular_range'],
                          PROJECTION_PARA['source_detector_distance'], PROJECTION_PARA['source_isocenter_distance'])
GEOMETRY.set_projection_matrices(circular_trajectory.circular_trajectory_3d(GEOMETRY))