import json
import torch

"""
# Load the combined dataset
with open('/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/combined_annotated_august_prostates.json', 'r') as f:
    data = json.load(f)

# Load the annotated_slices_per_patient
with open('/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/annotated_slices_per_patient.json', 'r') as f:
    annotated_slices_per_patient = json.load(f)

# Filter out patients with 3 or fewer annotated slices directly in the data list
print("Starting the filtering process...")
initial_len = len(data)
data = [patient for patient in data if len(annotated_slices_per_patient.get(patient['Folder_id'], [])) > 3]
print(f"Initial length: {initial_len}")
print(f"Final length: {len(data)}")

# Print how many patients were removed
print(f"Removed {initial_len - len(data)} patients with 3 or fewer annotated slices.")

# Save the filtered data
print("Saving the filtered data...")
with open('/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/filtered_august_prostates.json', 'w') as f:
    json.dump(data, f)

print("Filtering and saving complete!")
"""

import json

# Load the filtered data
with open('/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/filtered_august_prostates.json', 'r') as f:
    data = json.load(f)

def hard_filtering(data):
    filtered_data = []

    for patient in data:
        patient_id = patient['Folder_id']
        print(f"Processing patient {patient_id}...")
        
        # Extract the slice number from the slice path
        slice_number = patient['slice'].split('_')[-1].replace('.dcm', '')
        slice_number = int(slice_number.split('/')[-1])
        print(f"Slice number: {slice_number}")
        
        # Check if the label contains any non-zero values (i.e., annotated pixels)
        if not any(pixel > 0 for pixel in patient['label']):
            print(f"Label for slice {slice_number} contains only zeros, skipping...")
            continue
        else: 
            print(f"Label for slice {slice_number} contains non-zero values, including in the filtered data...")
            # If the slice is annotated and has a valid label, include it in the filtered data
            filtered_data.append({
                'slice': patient['slice'],
                'label': patient['label'],
                'Folder_id': patient['Folder_id']
            })

    return filtered_data

# Apply hard filtering
print("Starting hard filtering of non-annotated slices...")
hard_filtered_data = hard_filtering(data)

# Save the hard filtered data
output_path = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/hard_filtered_annotates_prostates.json'
print(f"Saving the hard filtered data to {output_path}...")
with open(output_path, 'w') as f:
    json.dump(hard_filtered_data, f)

print("Hard filtering and saving complete!")