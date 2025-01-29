#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the constants that are used in the moosez.

.. moduleauthor:: Lalith Kumar Shiyam Sundar <lalith.shiyamsundar@meduniwien.ac.at>
.. versionadded:: 3.0.0
"""

VERSION = "3.1.0"

ALLOWED_MODALITIES = ['CT', 'PT', 'MR']

# COLOR CODES
ANSI_ORANGE = '\033[38;5;208m'
ANSI_GREEN = '\033[38;5;40m'
ANSI_VIOLET = '\033[38;5;141m'
ANSI_RED = '\033[38;5;196m'
ANSI_RESET = '\033[0m'

# FOLDER NAMES
SEGMENTATIONS_FOLDER = 'segmentations'
STATS_FOLDER = 'stats'

# PREPROCESSING PARAMETERS
INTERPOLATION = 'bspline'
CHUNK_THRESHOLD_RESAMPLING = 150
CHUNK_THRESHOLD_INFERRING = 350
OVERLAP_PER_AXIS = (0, 20, 20, 20)

# MODELS
KEY_FOLDER_NAME = "folder_name"
KEY_URL = "url"
KEY_LIMIT_FOV = "limit_fov"
KEY_DESCRIPTION = "description"
KEY_DESCRIPTION_TEXT = "Tissue of Interest"
KEY_DESCRIPTION_MODALITY = "Modality"
KEY_DESCRIPTION_IMAGING = "Imaging"
DEFAULT_SPACING = (1.5, 1.5, 1.5)
FILE_NAME_DATASET_JSON = "dataset.json"
FILE_NAME_PLANS_JSON = "plans.json"


ENHANCE_URL = "https://enhance-pet.s3.eu-central-1.amazonaws.com/enhance-pet-1_6k/ENHANCE-PET-1_6k.zip"

USAGE_MESSAGE = """
Usage:
  moosez -d <MAIN_DIRECTORY> -m <MODEL_NAMES> -b
Example:  
  moosez -d /Documents/Data_to_moose/ -m clin_ct_organs

Description:
  MOOSE (Multi-organ objective segmentation) - A data-centric AI solution that
  generates multilabel organ segmentations for systemic TB whole-person research."""

"""

This module contains the constants that are used in the moosez.

Constants are values that are fixed and do not change during the execution of a program. 
They are used to store values that are used repeatedly throughout the program, such as 
file paths, folder names, and display parameters.

This module contains the following constants:

- `ALLOWED_MODALITIES`: A constant that stores a list of allowed modalities for the moosez algorithm.

- `ANSI_ORANGE`: A constant that stores the ANSI color code for orange.
- `ANSI_GREEN`: A constant that stores the ANSI color code for green.
- `ANSI_VIOLET`: A constant that stores the ANSI color code for violet.
- `ANSI_RESET`: A constant that stores the ANSI color code for resetting the color.

- `SEGMENTATIONS_FOLDER`: A constant that stores the name of the folder that contains the segmentations generated by 
                          the moosez algorithm.
- `STATS_FOLDER`: A constant that stores the name of the folder that contains the statistics generated by 
                  the moosez algorithm.

- `INTERPOLATION`: A constant that stores the interpolation method used by the moosez algorithm.
- `CHUNK_THRESHOLD`: A constant that stores the chunk threshold used by the moosez algorithm.

This module is imported by other modules in the moosez package and the constants are used throughout the package 
to provide fixed values that are used repeatedly.
"""
