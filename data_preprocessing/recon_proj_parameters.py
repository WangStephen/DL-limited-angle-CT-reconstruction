import numpy as np


####################################### Projection parameters
# Declare Parameters
PROJECTION_PARA = {}

# Detector Parameters:
PROJECTION_PARA['detector_shape'] = [256, 256]
PROJECTION_PARA['detector_spacing'] = [2.6, 2.6]

# Trajectory Parameters:
PROJECTION_PARA['number_of_projections'] = 361
PROJECTION_PARA['angular_range'] = np.radians(360)

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