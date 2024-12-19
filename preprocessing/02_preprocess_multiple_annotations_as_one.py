"""

This script takes the filtered dataset (output from Step 01) and performs DICOM loading, label extraction, and segmentation mask creation. 
Merge multiple annotation labels into a single binary label (e.g., group all PIRADS≥3 lesions into one “cancer” label).

"""

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
        dicom_folder (str): The path to the folder containing DICOM files (coming from the excel file).
    """
    if not os.path.exists(dicom_folder):
        print(f"Folder {dicom_folder} does not exist")
        # pass this folder
        return []
    
    dicom_files = [pydicom.dcmread(os.path.join(dicom_folder, f)) for f in sorted(os.listdir(dicom_folder)) if f.endswith('.dcm')]
    dicom_files = sorted(dicom_files, key=lambda x: x.InstanceNumber)
    return dicom_files

def load_segmentation_masks(segmentation_json_path):
    """
    Load segmentation masks from a JSON file.

    Parameters:
        segmentation_json_path (str): The path to the JSON file containing the segmentation masks.
        In our case now, this would be .json
    """
    with open(segmentation_json_path, 'r') as f:
        df = pd.read_json(f)
    segmentation_data = df.loc[df['type'] == 'table', 'data'].values[0]
    return segmentation_data

def create_mask_from_data(data, shape=(512, 512)):
    """
    Since the segmentations are stored as a list of pixel indices,
    we need to convert them into a binary mask.
    For that we create a mask with zeros and then set the pixels
    in the data list to 1.
    """
    mask = np.zeros(shape)
    mask = mask.flatten()
    mask[data] = 1
    return mask

def load_patient_folder_mapping(excel_path):
    """
    Load the mapping between DICOM folders, patient IDs, and the modality folder from an Excel file.

    Args:
        excel_path (str): The path to the Excel file containing the mapping.
    
    Returns:
        dict: A dictionary with DICOM folders as keys and a list of dictionaries containing patient IDs, zona, and modality as values.
    """
    df = pd.read_excel(excel_path)
    print(f"Number of rows in the Excel file: {len(df)}")

    df['PATIENT (dicom_file_id)'] = df['PATIENT (dicom_file_id)'].astype(str).str.strip()
    df['FOLDER (dicom patient)'] = df['FOLDER (dicom patient)'].astype(str).str.strip()
    df['FOLDER (dicom labeled)'] = df['FOLDER (dicom labeled)'].astype(int)
    df['zona'] = df['zona'].astype(str).str.strip()

    # Initialize the mapping dictionary
    mapping = {}

    for _, row in df.iterrows():
        folder_name = str(row['FOLDER (dicom patient)']).strip()
        patient_id = str(row['PATIENT (dicom_file_id)']).strip()
        zona = str(row['zona']).strip()
        modality = row['FOLDER (dicom labeled)']

        entry = {
            'patient_id': patient_id,
            'zona': zona,
            'modality': modality
        }

        if folder_name not in mapping:
            mapping[folder_name] = [entry]
        else:
            # Append the new entry to the list
            mapping[folder_name].append(entry)

    # Count the total number of entries
    total_entries = sum(len(v) for v in mapping.values())

    print(f"Number of patients in the mapping: {total_entries}")
    print(f"Mapping: {mapping}")
    return mapping

def create_json_file_for_patient(dicom_folder, patient_data, folder_name, zona): 
    """
    This function will create JSON entries for each DICOM slice of a patient.

    Args:
        dicom_folder (str): The path to the folder containing DICOM files.
        patient_data (dict): The data for the patient.
        folder_name (str): The folder name corresponding to the patient.
        modality (int): The modality that has been annotated.
        zona (str): The zona that has been annotated.

    Returns:
        list: A list of dictionaries containing the slice paths, segmentation masks, and folder names.
    """
    dicom_files = load_dicom_files(dicom_folder)
    print(f"Number of DICOM files: {len(dicom_files)}")

    labelmap_keys = ['labelmap3', 'labelmap4']  # Adjust this list as needed
    
    masks = {}
    for labelmap_key in labelmap_keys:
        segmentation_masks = patient_data.get(labelmap_key, [])
        
        # Parse the segmentation masks if they are strings
        if isinstance(segmentation_masks, str):
            try:
                segmentation_masks = ast.literal_eval(segmentation_masks)
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing segmentation masks for {labelmap_key}: {e}")
                continue  # Skip this labelmap_key if parsing fails

        print(f"Number of segmentation masks in {labelmap_key}: {len(segmentation_masks)}")
        
        for mask in segmentation_masks:
            # Ensure mask is a list and has at least one element
            if not isinstance(mask, list) or len(mask) == 0:
                print(f"Invalid mask format in {labelmap_key}: {mask}")
                continue  # Skip invalid masks

            segmentation_mask_dict = mask[0]
            layer = segmentation_mask_dict.get('layer')
            data = segmentation_mask_dict.get('data')

            if layer is None or data is None:
                print(f"Missing 'layer' or 'data' in segmentation mask: {segmentation_mask_dict}")
                continue  # Skip masks missing required fields

            # Parse 'data' if it's a string
            if isinstance(data, str):
                try:
                    data = ast.literal_eval(data)
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing 'data' for layer {layer} in {labelmap_key}: {e}")
                    continue  # Skip this mask if parsing fails

            # Ensure 'data' is a list of integers
            if not isinstance(data, list) or not all(isinstance(i, int) for i in data):
                print(f"'data' is not a list of integers for layer {layer} in {labelmap_key}: {data}")
                continue  # Skip invalid 'data'

            # Initialize the set for the layer if it doesn't exist
            if layer not in masks:
                masks[layer] = set()

            # Update the mask with the new data
            masks[layer].update(data)
    
    data = []
    count = 0
    for dicom in dicom_files:
        image_path = os.path.join(dicom_folder, dicom.filename)
        mask = np.zeros((512, 512)).flatten()
        layer = dicom.InstanceNumber

        if layer in masks:
            count += 1
            segmentation_mask = list(masks[layer])  
            mask = create_mask_from_data(segmentation_mask)
        
        entry = {
            'slice': image_path,
            'label': mask.tolist(),
            'Folder_id': folder_name,  
            'Zona': zona
        }
        data.append(entry)
    
    print(f"Processed folder {folder_name}, zone {zona}, where number of slices with segmentation masks: {count}")
    return data


def process_all_patients(unfiltered_json_path, excel_path, dicom_base_dir, output_path):
    """
    Process all patients in the dataset and create a combined JSON file.

    Args:
        unfiltered_json_path (str): The path to the JSON file containing the segmentation masks.
        excel_path (str): The path to the Excel file containing the patient-DICOM folder mapping.
        dicom_base_dir (str): The base directory containing the DICOM folders.
        output_path (str): The path to save the combined JSON file.
    """
    segmentation_data = load_segmentation_masks(unfiltered_json_path)
    folder_mapping = load_patient_folder_mapping(excel_path)

    combined_data_dict = {}
    missing_folders = []

    for folder in folder_mapping:
        print(f"\nProcessing folder: {folder}")
        entries = folder_mapping[folder]
        print(f"Number of entries in folder mapping: {len(entries)}")

        for entry in entries:
            patient_id = entry['patient_id']
            modality = entry['modality']
            zona = entry['zona']

            print(f"\nProcessing entry - Patient ID: {patient_id}, Modality: {modality}, Zona: {zona}")

            # Filter modalities different from 5
            if modality != 5:
                print(f"Modality {modality} is not 5. Skipping this entry.")
                continue

            modality_str = str(modality).strip()

            # Filter segmentation_data without considering 'zona'
            patient_data = [
                data for data in segmentation_data 
                if data['dicom_file_id'].strip() == patient_id 
                and data['folder'].strip() == modality_str 
                and data['zona'].strip() == zona
            ]

            if not patient_data:
                print(f"No data found for Patient ID: {patient_id} with Modality: {modality_str} and Zona: {zona}")
                missing_folders.append(folder)
                continue
            else:
                print(f"Found data for Patient ID: {patient_id}, Modality: {modality_str}, Zona: {zona}")
                # print(f"Number of entries in patient data: {len(patient_data)}")
                print(f"Keys: {patient_data[0].keys()}")
                patient_data = patient_data[0]
            
            if modality == 5:
                dicom_folder = os.path.join(dicom_base_dir, folder, 'Ax_T2_frFSE')
            else:
                # print(f"Unknown modality {modality} for folder {folder}")
                print(f"Modality {modality} not T2, skipping...")
                continue
            # elif modality == 13:
                # dicom_folder = os.path.join(dicom_base_dir, folder, 'ADC_(10_-6_mm2_s):_DATE_')
                # print(f"Modality 13 = ADC_(10_-6_mm2_s):_DATE_, skipping...")
                # continue
            # elif modality == 4:
                # dicom_folder = os.path.join(dicom_base_dir, folder, 'Ax__Diff_b1000_PELVIS')
                # print(f"Modality 4 = Ax__Diff_b1000_PELVIS, skipping...")
                # continue 

            # Create JSON entries for the patient
            patient_data_entries = create_json_file_for_patient(dicom_folder, patient_data, folder, zona)
            print(f"Number of entries in patient data entries: {len(patient_data_entries)}")

            for entry_data in patient_data_entries:
                # Remove 'Zona' from entry since we are merging annotations across zones
                if 'Zona' in entry_data:
                    del entry_data['Zona']
                
                # Create a unique key based on 'slice' and 'Folder_id' to identify unique slices
                slice_id = entry_data.get('slice')
                folder_id = entry_data.get('Folder_id')

                if slice_id is None or folder_id is None:
                    print(f"Entry missing 'slice' or 'Folder_id' key: {entry_data}")
                    continue

                key = (slice_id, folder_id)  # Tuple ensures uniqueness based on slice and folder

                if key in combined_data_dict:
                    # If the slice is already in the dictionary, merge the labels
                    print(f"The slice {slice_id} in folder {folder_id} is already in the dictionary")
                    print("We merge the new label with the existing label (keeping the maximum, therefore the annotated one)")
                    
                    existing_label = np.array(combined_data_dict[key]['label'])
                    print(f"The existing label has unique values: {np.unique(existing_label)}")
                    new_label = np.array(entry_data['label'])
                    print(f"The new label has unique values: {np.unique(new_label)}")
                    
                    # Merge labels by taking the maximum value for each pixel
                    merged_label = np.maximum(existing_label, new_label)
                    # maybe this needs to be change to taking the union of the existing and new, i need to understand why they change zone but not slice.
                    combined_data_dict[key]['label'] = merged_label.tolist()
                    
                    print(f"Merged label for slice {slice_id} in folder {folder_id}")
                    print(f"The merged label has unique values: {np.unique(merged_label)}")
                else:
                    # If the slice is not in the dictionary, add it
                    #print(f"The slice {slice_id} in folder {folder_id} is not in the dictionary")
                    combined_data_dict[key] = entry_data
                    print(f"Added slice {slice_id} in folder {folder_id} to the combined data dictionary.")
        
    # Convert the combined data dictionary to a list
    combined_data = list(combined_data_dict.values())

    # Counting total entries in the combined data
    total_entries = len(combined_data)
    print(f"\nTotal number of entries in combined_data: {total_entries}")
    
    # Counting slices with labels (regardless of duplicates)
    count_slices_with_labels = sum(
        1 for entry in combined_data if any(value != 0.0 for value in entry['label'])
    )
    print(f"Total number of slices with labels: {count_slices_with_labels}")
    
    # Save the combined data to a JSON file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(combined_data, f)
    
    print(f"Combined JSON file saved to {output_path}")
    print(f"Missing folders: {missing_folders}")

# ------------------------------- Define the paths and process all patients -------------------------------

unfiltered_json_path = 'dataset/13_12_2024_finalversion/all_annotated_patients_december_2024.json'
excel_path = 'dataset/13_12_2024_finalversion/all_annotated_patients_december_2024.xlsx'
dicom_base_dir = '/ssds/DATA/Prostate/MRI/FILTERED_HOSP'
output_path = 'dataset/combined_annotated_december_cancer.json'

process_all_patients(unfiltered_json_path, excel_path, dicom_base_dir, output_path)