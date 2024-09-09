#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------------
# Author: Lalith Kumar Shiyam Sundar
# Institution: Medical University of Vienna
# Research Group: Quantitative Imaging and Medical Physics (QIMP) Team
# Date: 13.02.2023
# Version: 2.0.0
#
# Description:
# This module contains the necessary functions for prediction using the moosez.
#
# Usage:
# The functions in this module can be imported and used in other modules within the moosez to perform prediction.
#
# ----------------------------------------------------------------------------------------------------------------------

import os
import dask.array as da
import dask
import sys
import torch
import numpy as np
from moosez import constants
from moosez import image_processing
from moosez.resources import MODELS, AVAILABLE_MODELS, check_device

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def initialize_predictor(model_name: str, accelerator: str) -> nnUNetPredictor:
    """
    Initializes the model for prediction.

    :param model_name: The name of the model.
    :type model_name: str
    :param accelerator: The accelerator for prediction.
    :type model_name: str
    :return: The initialized predictor object.
    :rtype: nnUNetPredictor
    """
    model_folder_name = MODELS[model_name]["directory"]
    trainer = MODELS[model_name]["trainer"]
    configuration = MODELS[model_name]["configuration"]
    planner = MODELS[model_name]["planner"]
    device = torch.device(accelerator)
    predictor = nnUNetPredictor(allow_tqdm=False, device=device)
    predictor.initialize_from_trained_model_folder(os.path.join(constants.NNUNET_RESULTS_FOLDER, model_folder_name, f"{trainer}__{planner}__{configuration}"), use_folds=("all", ))
    return predictor


@dask.delayed
def process_case(preprocessor, chunk: np.ndarray, chunk_properties: dict, predictor, location: tuple):
    data, seg = preprocessor.run_case_npy(chunk,
                                          None,
                                          chunk_properties,
                                          predictor.plans_manager,
                                          predictor.configuration_manager,
                                          predictor.dataset_json)
    return {'data': torch.from_numpy(data).contiguous().pin_memory(), 'data_properties': chunk_properties, 'ofile': None,
            'location': location}


def preprocessing_iterator_from_dask_array(dask_array: da.Array, image_properties: dict, predictor):
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=predictor.verbose)

    chunk_indices = []
    delayed_chunks = dask_array.to_delayed()
    delayed_tasks = []

    for chunk_index in np.ndindex(delayed_chunks.shape):
        chunk_indices.append(chunk_index)
        image_chunk = delayed_chunks[chunk_index]
        delayed_task = dask.delayed(process_case)(preprocessor, image_chunk, image_properties, predictor, chunk_index)
        delayed_tasks.append(delayed_task)

    results = dask.compute(*delayed_tasks)
    iterator = iter(results)

    return iterator, chunk_indices


def reconstruct_array_from_chunks(chunks: list[np.array], chunk_positions: list[tuple], original_shape: tuple):
    reconstructed_array = np.empty(original_shape, dtype=chunks[0].dtype)

    for chunk, position in zip(chunks, chunk_positions):
        chunk = chunk[None, ...]
        slices = tuple(slice(pos * csize, pos * csize + csize) for pos, csize in zip(position, chunk.shape))
        reconstructed_array[slices] = chunk

    return np.squeeze(reconstructed_array)


def predict_from_array_by_iterator(image_array: np.ndarray, model_name: str, accelerator: str = None):
    image_array = image_array[None, ...]
    chunks = [axis / image_processing.ImageResampler.chunk_along_axis(axis) for axis in image_array.shape]
    prediction_array = da.from_array(image_array, chunks=chunks)

    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull

        try:
            if accelerator is None:
                accelerator = check_device()
            predictor = initialize_predictor(model_name, accelerator)
            image_properties = {
                'spacing': MODELS[model_name]["voxel_spacing"]
            }
            iterator, chunk_locations = preprocessing_iterator_from_dask_array(prediction_array, image_properties,
                                                                               predictor)
            segmentations = predictor.predict_from_data_iterator(iterator)
            combined_segmentations = reconstruct_array_from_chunks(segmentations, chunk_locations, prediction_array.shape)
            return combined_segmentations

        finally:
            sys.stdout = old_stdout


def construct_prediction_routine(models: str | list[str]) -> dict[tuple, list[str]]:
    if isinstance(models, str):
        models = [models]

    prediction_routine = {}
    for model in models:
        if model in AVAILABLE_MODELS:
            model_spacing = tuple(MODELS[model]["voxel_spacing"])
            if model_spacing in prediction_routine:
                prediction_routine[model_spacing].append(model)
            else:
                prediction_routine[model_spacing] = [model]

    return prediction_routine
