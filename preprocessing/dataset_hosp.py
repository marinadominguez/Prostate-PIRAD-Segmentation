"""

This dataloader reads from a combined JSON file containing slice-level data for multiple patients. 
It groups the slices by patient, forming a 3D volume (H x W x D) and a corresponding label volume. 
The dataset is then split into train, validation, and test sets.
The code allows specifying transformations for data augmentation or normalization. 
It also accommodates both automatically generated splits (via configuration parameters) and manually specified patient splits.

"""

import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pydicom
import HOSP_pipeline.config.config as cfg

import sys
import os

# Add the project directory (HOSP_pipeline) to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class Prostate3DDataset(Dataset):
    def __init__(self, data, transforms=None):
        """
        Args:
            json_file (str): Path to the combined JSON file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.patients = self.group_slices_by_patient()
        self.transforms = transforms

    def group_slices_by_patient(self):
        """
        Groups slices by patient ID and converts them into 3D volumes.
        
        Returns:
            dict: A dictionary with patient IDs as keys and 3D volumes as values.
        """
        patient_dict = {}
        sorted_data = sorted(self.data, key=lambda x: (x['Folder_id'], x['slice'], x['label']))
        # Sort the data based on patient ID and slice number
        for entry in sorted_data:
            patient_id = entry['Folder_id']
            # print(f"Processing patient: {patient_id}") # Processing patient: cd2eaece70fbd0f
            slice_path = entry['slice']
            # print(f"Slice path: {slice_path}") # Slice path: /ssds/DATA/Prostate/MRI/FILTERED_HOSP/cd2eaece70fbd0f/Ax_T2_frFSE/0170.dcm
            label = np.array(entry['label']).reshape((512, 512))
            # print(f"Label shape: {label.shape}") # Label shape: (512, 512) 
            
            if patient_id not in patient_dict:
                patient_dict[patient_id] = {'image': [], 'label': []}
            
            # Read the slice image and append it
            slice_img = pydicom.dcmread(slice_path).pixel_array
            # print(f"Slice image shape: {slice_img.shape}") # Slice image shape: (512, 512)
            patient_dict[patient_id]['image'].append(slice_img)
            patient_dict[patient_id]['label'].append(label)  
            # print(f"Patient dictionary length: {len(patient_dict)}") # Patient dictionary length: 17        
        
        # Convert lists to numpy arrays (3D volumes)
        for patient_id in patient_dict:
            patient_dict[patient_id]['image'] = np.stack(patient_dict[patient_id]['image'])
            patient_dict[patient_id]['label'] = np.stack(patient_dict[patient_id]['label'])
        
        print(f"Patient dictionary length: {len(patient_dict)}")
        print(f"Patient dictionary keys: {list(patient_dict.keys())}")
        print(f"For the patient {list(patient_dict.keys())[0]}, the image shape is: {patient_dict[list(patient_dict.keys())[0]]['image'].shape}")
        print(f"For the patient {list(patient_dict.keys())[0]}, the label shape is: {patient_dict[list(patient_dict.keys())[0]]['label'].shape}")
        return patient_dict

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id = list(self.patients.keys())[idx]
        sample = self.patients[patient_id]

        if self.transforms:
            sample = self.transforms(sample)

        return sample['image'], sample['label']

def get_train_val_test_Dataloaders(train_transforms, val_transforms, test_transforms, train_ids=None, val_ids=None, test_ids=None):
    """
    Creates train, validation, and test dataloaders from the dataset.

    Args:
        train_transforms: Transformations to apply to training data.
        val_transforms: Transformations to apply to validation data.
        test_transforms: Transformations to apply to test data.
        train_ids (list, optional): List of patient IDs to use for training.
        val_ids (list, optional): List of patient IDs to use for validation.
        test_ids (list, optional): List of patient IDs to use for testing.

    Returns:
        DataLoader: Train DataLoader
        DataLoader: Validation DataLoader
        DataLoader: Test DataLoader
    """
    # Load the data from JSON
    with open(cfg.DATASET_PATH, "r") as file:
        data = json.load(file)

    # Group data by patient ID
    patients_data = {}
    for entry in data:
        patient_id = entry['Folder_id']
        if patient_id not in patients_data:
            patients_data[patient_id] = []
        patients_data[patient_id].append(entry)

    # Create list of all patient IDs
    all_patient_ids = list(patients_data.keys())
    print(f"All patient IDs in dataset: {all_patient_ids}")

    # If IDs are provided, use them
    if train_ids is not None and val_ids is not None and test_ids is not None:
        # Ensure that the provided IDs are in the dataset
        train_ids = [pid for pid in train_ids if pid in all_patient_ids]
        val_ids = [pid for pid in val_ids if pid in all_patient_ids]
        test_ids = [pid for pid in test_ids if pid in all_patient_ids]

        # Check for any IDs not found
        missing_ids = set(train_ids + val_ids + test_ids) - set(all_patient_ids)
        if missing_ids:
            print(f"The following IDs are not present in the dataset: {missing_ids}")

        # Ensure no overlap between sets
        overlap = set(train_ids) & set(val_ids) | set(train_ids) & set(test_ids) | set(val_ids) & set(test_ids)
        if overlap:
            raise ValueError(f"Overlap found in train, val, and test IDs: {overlap}")
    else:
        # If IDs are not provided, perform random split
        np.random.seed(cfg.SPLIT_SEED)
        np.random.shuffle(all_patient_ids)

        num_patients = len(all_patient_ids)
        num_test = max(1, int(num_patients * cfg.TRAIN_VAL_TEST_SPLIT[2]))
        num_val = max(1, int(num_patients * cfg.TRAIN_VAL_TEST_SPLIT[1]))
        num_train = max(1, num_patients - num_val - num_test)

        test_ids = all_patient_ids[:num_test]
        val_ids = all_patient_ids[num_test:num_test + num_val]
        train_ids = all_patient_ids[num_test + num_val:]

    # Print split info
    print(f"Number of patients in train: {len(train_ids)}")
    print(f"Number of patients in validation: {len(val_ids)}")
    print(f"Number of patients in test: {len(test_ids)}")

    print(f"Train IDs: {sorted(train_ids)}")
    print(f"Validation IDs: {sorted(val_ids)}")
    print(f"Test IDs: {sorted(test_ids)}")

    # Creating data lists for each split
    train_data = [item for pid in train_ids for item in patients_data[pid]]
    val_data = [item for pid in val_ids for item in patients_data[pid]]
    test_data = [item for pid in test_ids for item in patients_data[pid]]

    # Initialize datasets
    train_dataset = Prostate3DDataset(train_data, transforms=train_transforms)
    val_dataset = Prostate3DDataset(val_data, transforms=val_transforms)
    test_dataset = Prostate3DDataset(test_data, transforms=test_transforms)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.VAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.TEST_BATCH_SIZE, shuffle=False)

    print(f"Train loader size: {len(train_loader)}")
    print(f"Validation loader size: {len(val_loader)}")
    print(f"Test loader size: {len(test_loader)}")

    return train_loader, val_loader, test_loader