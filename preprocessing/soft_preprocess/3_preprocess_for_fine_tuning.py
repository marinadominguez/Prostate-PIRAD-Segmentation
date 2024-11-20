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
with open('/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/preprocessing/soft_preprocess/filtered_august_prostates.json', 'w') as f:
    json.dump(data, f)

print("Filtering and saving complete!")
"""

import json

# Load the filtered data
with open('/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/preprocessing/soft_preprocess/filtered_august_prostates.json', 'r') as f:
    data = json.load(f)

def soft_filtering(data):
    filtered_data = []

    # Group slices by patient
    patients_dict = {}
    for entry in data:
        patient_id = entry['Folder_id']
        if patient_id not in patients_dict:
            patients_dict[patient_id] = []
        patients_dict[patient_id].append(entry)

    # Apply filtering
    for patient_id, slices in patients_dict.items():
        print(f"Processing patient {patient_id}...")
        
        # Sort slices by their slice number
        slices = sorted(slices, key=lambda x: int(x['slice'].split('_')[-1].replace('.dcm', '').split('/')[-1]))
        
        # Determine the first 5 and last 5 slices
        first_5_slices = slices[:5]
        last_5_slices = slices[-5:]
        
        # Collect all annotated slices
        annotated_slices = [slice_data for slice_data in slices if any(pixel > 0 for pixel in slice_data['label'])]
        
        # Add first 5, last 5, and annotated slices to the final list
        slices_to_keep = first_5_slices + last_5_slices + annotated_slices
        
        # Remove duplicates while maintaining order
        seen_slices = set()
        for slice_data in slices_to_keep:
            slice_identifier = (slice_data['Folder_id'], slice_data['slice'])
            if slice_identifier not in seen_slices:
                filtered_data.append(slice_data)
                seen_slices.add(slice_identifier)
                print(f"Including slice {slice_data['slice']} in filtered data.")
            else:
                print(f"Duplicate slice {slice_data['slice']} skipped.")

    return filtered_data

# Apply soft filtering
print("Starting soft filtering of non-annotated slices...")
soft_filtered_data = soft_filtering(data)

# Save the soft filtered data
output_path = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/preprocessing/soft_preprocess/soft_filtered_annotated_prostates.json'
print(f"Saving the soft filtered data to {output_path}...")
with open(output_path, 'w') as f:
    json.dump(soft_filtered_data, f)

print("Soft filtering and saving complete!")