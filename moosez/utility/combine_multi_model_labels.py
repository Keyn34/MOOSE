from moosez import models
from moosez import image_processing
from moosez import system
from moosez import file_utilities
import SimpleITK
import os
import json


def create_multi_model_label_segmentation(segmentation_directory: str, model_identifiers: list[str]) -> (SimpleITK.Image, dict):
    output_manager = system.OutputManager(True, False)
    prepared_models = [models.Model(model_identifier, output_manager) for model_identifier in model_identifiers]

    base_segmentation_file_path = file_utilities.get_files(segmentation_directory, prepared_models[0].multilabel_prefix, ('.nii', '.nii.gz'))[0]
    combined_segmentation_image = SimpleITK.ReadImage(base_segmentation_file_path)
    combined_segmentation_dict = prepared_models[0].organ_indices
    output_manager.console_update(f"Starting with {prepared_models[0]} as base.")
    for prepared_model in prepared_models[1:]:
        output_manager.console_update(f"  - combining with {prepared_model}")
        new_segmentation_file_path = file_utilities.get_files(segmentation_directory, prepared_model.multilabel_prefix, ('.nii', '.nii.gz'))[0]
        new_segmentation_image = SimpleITK.ReadImage(new_segmentation_file_path)
        combined_segmentation_image, combined_segmentation_dict = image_processing.add_model_label_image(combined_segmentation_image, combined_segmentation_dict, new_segmentation_image, prepared_model.organ_indices)

    combined_segmentation_file = os.path.join(segmentation_directory, f"combined_segmentations.nii.gz")
    SimpleITK.WriteImage(combined_segmentation_image, combined_segmentation_file)

    combined_segmentation_json_file = os.path.join(segmentation_directory, f"combined_segmentations.json")
    with open(combined_segmentation_json_file, "w") as json_file:
        json.dump(combined_segmentation_dict, json_file, indent=4)

    return combined_segmentation_image, combined_segmentation_dict


if __name__ == '__main__':
    model_identifiers = ["clin_ct_organs", "clin_ct_peripheral_bones", "clin_ct_vertebrae", "clin_ct_muscles", "clin_ct_ribs"]
    segmentation_directory_path = "/home/horyzen/Downloads/MOOSE/multi-model-labels/2_vision_1/moosez-2024-11-02-19-40-21/segmentations"
    _, label_organ_indices = create_multi_model_label_segmentation(segmentation_directory_path, model_identifiers)
    for index, label in label_organ_indices.items():
        print(f"{index}: {label}")
