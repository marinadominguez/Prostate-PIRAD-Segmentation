import os
import json
import numpy as np
import pydicom
import ast
import pandas as pd

def load_dicom_files(dicom_folder):
    """
    Load DICOM files from a specified folder.

    Args:
        dicom_folder (str): The path to the folder containing DICOM files.

    Returns:
        list: A list of DICOM files sorted by their InstanceNumber.
    """
    dicom_files = [pydicom.dcmread(os.path.join(dicom_folder, f)) for f in sorted(os.listdir(dicom_folder)) if f.endswith('.dcm')]
    dicom_files = sorted(dicom_files, key=lambda x: x.InstanceNumber)
    return dicom_files

def load_segmentation_masks(segmentation_json_path):
    """
    Load segmentation masks from a JSON file.

    Parameters:
    segmentation_json_path (str): The path to the JSON file containing the segmentation masks.

    Returns:
    dict: A dictionary containing the loaded segmentation masks for the first patient.
    """
    with open(segmentation_json_path, 'r') as f:
        df = pd.read_json(f)
        print(df)
    # Extract the relevant segmentation data
    segmentation_data = df.loc[df['type'] == 'table', 'data'].values[0]
    print(f"Number of patients in segmentation data: {len(segmentation_data)}")
    # Select the first patient 
    # CHANGE THIS TO CHANGE THE PATIENT ON THE DOWNLOADED JSON FILE
    patient_data = segmentation_data[0]
    print(f"Patient data keys: {patient_data.keys()}")
    # Extract the segmentation data from 'labelmap1'
    segmentation_masks = patient_data['labelmap1']
    # Segmentation mask is now a string, so we need to convert it to a dictionary
    segmentation_masks = ast.literal_eval(segmentation_masks)
    print(f"Segmentation mask length: {len(segmentation_masks)}")
    masks = {}
    for i in range(len(segmentation_masks)):
        segmentation_mask_dict = segmentation_masks[i][0]
        layer = segmentation_mask_dict['layer']
        data = segmentation_mask_dict['data']
        masks[layer] = data
    print(f"Number of segmentation masks: {len(masks)}")
    print(f"Segmentation mask keys: {masks.keys()}")
    return masks

def create_mask_from_data(data, shape=(512, 512)):
    """
    Create a binary mask from the segmentation data.
    Since the segmentations are stored as a list of pixel indices, 
    we need to convert them to a binary mask.

    Args:
        data (list): The list of pixel indices.
        shape (tuple): The shape of the mask.
    
    Returns:
        numpy.ndarray: The binary mask
    """
    mask = np.zeros(shape)
    mask = mask.flatten()
    mask[data] = 1
    return mask

def create_json_file(dicom_folder, segmentation_json_path, output_path):
    # Load and sort the DICOM files
    dicom_files = load_dicom_files(dicom_folder)
    num_slice_files = len(dicom_files)
    print(f"number of slices in this folder {num_slice_files}")

    # Load the segmentation masks
    segmentation_masks = load_segmentation_masks(segmentation_json_path)

    # Create a list to store the image and label data
    data = []
    
    count = 0
    # Iterate over the DICOM files and corresponding label slices
    for i, dicom in enumerate(dicom_files):
        image_path = os.path.join(dicom_folder, dicom.filename)
        print(f"Processing slice {i+1}/{num_slice_files}")

        # Initialize the mask to zeros (no segmentation)
        mask = np.zeros((512, 512))
        mask = mask.flatten()

        layer = dicom.InstanceNumber
        # Check if there's a corresponding segmentation for the current slice
        if layer in segmentation_masks:
            print(f"Found segmentation mask for slice {i+1} (layer {layer})")
            count += 1
            # The segmentation mask is a string, so we need to convert it to a list
            segmentation_mask = ast.literal_eval(segmentation_masks[layer])
            print(f"Segmentation mask length: {len(segmentation_mask)}")
            mask = create_mask_from_data(segmentation_mask)
            print(f"Mask shape: {mask.shape}")
        else:
            print(f"No segmentation mask found for slice {i+1}")
            print(f"Creating empty mask for slice {i+1}")
            print(f"Mask shape: {mask.shape}")

        # Create a dictionary with the image path and label data
        entry = {
            'slice': image_path,
            'label': mask.tolist()  # Convert numpy array to list for JSON serialization
        }

        # Add the entry to the data list
        data.append(entry)
        print(f"Number of slices with segmentation masks: {count}")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write the data to a JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f)

# Example usage
dicom_folder = '/ssds/DATA/Prostate/MRI/FILTERED_HOSP/0f9cb0851d69ad4/Ax_T2_frFSE'
segmentation_json_path = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/preprocessing/first_one_by_one_approach/patient_11.json'
# '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/prostate_22_07_24.json'
output_path = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/prostate_pirad5_pat_11.json'
# '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/prostate_pat_25.json'

create_json_file(dicom_folder, segmentation_json_path, output_path)
