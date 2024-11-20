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

def get_train_val_test_Dataloaders(train_transforms, val_transforms, test_transforms):
    """
    Creates train, validation, and test dataloaders from the dataset.

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
    
    # Create list of data entries for easy access
    patient_ids = list(patients_data.keys())

    # Splitting patient IDs for train, validation, and test sets
    if len(patient_ids) < 3:
        raise ValueError("Not enough patients to split into train, val, and test sets.")

    np.random.seed(cfg.SPLIT_SEED)
    np.random.shuffle(patient_ids)
    
    # Calculate number of patients for each split, ensuring at least one patient per split
    num_patients = len(patient_ids)
    num_test = max(1, int(num_patients * cfg.TRAIN_VAL_TEST_SPLIT[2]))
    num_val = max(1, int(num_patients * cfg.TRAIN_VAL_TEST_SPLIT[1]))
    num_train = max(1, int(num_patients - num_val - num_test))

    train_ids = patient_ids[:num_train]
    val_ids = patient_ids[num_train:num_train + num_val]
    test_ids = patient_ids[num_train + num_val:]
    print(f"Number of patients in train: {len(train_ids)}")
    print(f"Number of patients in validation: {len(val_ids)}")
    print(f"Number of patients in test: {len(test_ids)}")

    # Print split info
    sorted_train_ids = sorted(train_ids)
    print(f"Train IDs: {sorted_train_ids}")
    #print(f"Validation IDs: {val_ids}")
    sorted_validation_ids = sorted(val_ids)
    print(f"Validation IDs: {sorted_validation_ids}")
    #print(f"Test IDs: {test_ids}")
    sorted_test_ids = sorted(test_ids)
    print(f"Test IDs: {sorted_test_ids}")

    # Creating data lists for each split
    train_data = [item for pid in sorted_train_ids for item in patients_data[pid]]
    val_data = [item for pid in sorted_validation_ids for item in patients_data[pid]]
    test_data = [item for pid in sorted_test_ids for item in patients_data[pid]]

    """ when too little data, this is not working
    # Split data into training, validation, and testing
    train_val_data, test_data = train_test_split(patient_ids, test_size=split_ratios[2], random_state=random_seed)
    train_data, val_data = train_test_split(train_val_data, test_size=split_ratios[1] / (split_ratios[0] + split_ratios[1]), random_state=random_seed)
    """

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

"""
# Example usage
train_loader, val_loader, test_loader = get_train_val_test_Dataloaders(None, None, None)

# Example iteration over the dataloaders
for batch in train_loader:
    images, labels = batch
    print(images.shape, labels.shape)
    break
"""