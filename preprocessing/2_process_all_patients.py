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
        In our case now, this would be 01_08_2024_PROSTATES.json
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
        dict: A dictionary with DICOM folders as keys and dictionaries containing patient IDs and modality as values.
    """
    df = pd.read_excel(excel_path)
    df['PATIENT (dicom_file_id)'] = df['PATIENT (dicom_file_id)'].astype(str).str.strip()  # Convert to string and remove spaces
    df['FOLDER (dicom patient)'] = df['FOLDER (dicom patient)'].astype(str).str.strip()  # Ensure folder names are strings
    
    # Create the mapping with folder names as keys
    mapping = {
        str(row['FOLDER (dicom patient)']).strip(): {
            'patient_id': str(row['PATIENT (dicom_file_id)']).strip(),
            'modality': row['FOLDER (dicom labeled)']
        }
        for _, row in df.iterrows()
    }
    
    print(f"Number of patients in the mapping: {len(mapping)}")
    print(f"Mapping: {mapping}")  # Print first 5 keys for debugging
    return mapping


def create_json_file_for_patient(dicom_folder, patient_data, folder_name):
    """
    This function will create JSON entries for each DICOM slice of a patient.

    Args:
        dicom_folder (str): The path to the folder containing DICOM files.
        patient_data (dict): The data for the patient.
        folder_name (str): The folder name corresponding to the patient.

    Returns:
        list: A list of dictionaries containing the slice paths, segmentation masks, and folder names.
    """
    dicom_files = load_dicom_files(dicom_folder)
    print(f"Number of DICOM files: {len(dicom_files)}")
    segmentation_masks = patient_data['labelmap1']
    segmentation_masks = ast.literal_eval(segmentation_masks)
    print(f"Number of segmentation masks: {len(segmentation_masks)}")
    
    masks = {}
    for i in range(len(segmentation_masks)):
        segmentation_mask_dict = segmentation_masks[i][0]
        layer = segmentation_mask_dict['layer']
        data = segmentation_mask_dict['data']
        masks[layer] = data
    
    data = []
    count = 0
    for dicom in dicom_files:
        image_path = os.path.join(dicom_folder, dicom.filename)
        #print(f"Image path: {image_path}")

        mask = np.zeros((512, 512)).flatten()
        layer = dicom.InstanceNumber
        #print(f"Processing slice {layer}")
        
        if layer in masks:
            count += 1
            segmentation_mask = ast.literal_eval(masks[layer])
            # print(f"Type of data in segmentation mask: {type(segmentation_mask)}") # list
            #print(f"Length of data in segmentation mask: {len(segmentation_mask)}")
            mask = create_mask_from_data(segmentation_mask)
        
        entry = {
            'slice': image_path,
            'label': mask.tolist(),
            'Folder_id': folder_name  # Use the folder name as the identifier
        }
        data.append(entry)
    
    print(f"Processed folder {folder_name}, number of slices with segmentation masks: {count}")
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

    combined_data = []
    missing_folders = []
    for folder in folder_mapping:
        print(f"Processing folder: {folder}")
        print(f"Values in folder mapping: {folder_mapping[folder]}")
        patient_id = folder_mapping[folder]['patient_id']
        #print(f"Patient ID: {patient_id}")
        modality = folder_mapping[folder]['modality']
        #print(f"Modality: {modality}")

        # Convert modality to string if necessary
        modality_str = str(modality).strip()

        patient_data = [
            data for data in segmentation_data 
            if data['dicom_file_id'].strip() == patient_id and data['folder'].strip() == modality_str
            ]

        if not patient_data:
            print(f"No data found for Patient ID: {patient_id} with Modality: {modality_str}")
        else:
            print(f"Found data for Patient ID: {patient_id}, Modality: {modality_str}")
            #print(f"Number of entries: {len(patient_data)}")
            #print(f"Keys: {patient_data[0].keys()}")
            patient_data = patient_data[0]
        
        if not patient_data:
            print(f"Folder {folder} not found in segmentation data")
            missing_folders.append(folder)
            continue
        
        if modality == 5:
            dicom_folder = os.path.join(dicom_base_dir, folder, 'Ax_T2_frFSE')
        elif modality == 13:
            # dicom_folder = os.path.join(dicom_base_dir, folder, 'ADC_(10_-6_mm2_s):_DATE_')
            print(f"Modality 13 = ADC_(10_-6_mm2_s):_DATE_, skipping...")
            continue
        elif modality == 4:
            # dicom_folder = os.path.join(dicom_base_dir, folder, 'Ax__Diff_b1000_PELVIS')
            print(f"Modality 4 = Ax__Diff_b1000_PELVIS, skipping...")
            continue
        else:
            print(f"Unknown modality {modality} for folder {folder}")
            continue

        patient_data_entries = create_json_file_for_patient(dicom_folder, patient_data, folder)
        combined_data.extend(patient_data_entries)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(combined_data, f)
    
    print(f"Combined JSON file saved to {output_path}")
    print(f"Missing folders: {missing_folders}")

unfiltered_json_path = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/24_10_2024/all_annotated_patients_october_2024.json'
#'/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/23_09_2024/all_annotated_patients_september_2024.json'
#'/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/29_08_2024/all_annotated_patients_august_2024.json'
excel_path = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/24_10_2024/all_annotated_patients_october_2024.xlsx'
# '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/23_09_2024/all_annotated_patients_september_2024.xlsx'
# '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/29_08_2024/prostates_08_2024.xlsx'
dicom_base_dir = '/ssds/DATA/Prostate/MRI/FILTERED_HOSP'
output_path = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/combined_annotated_october_prostates.json'

process_all_patients(unfiltered_json_path, excel_path, dicom_base_dir, output_path)