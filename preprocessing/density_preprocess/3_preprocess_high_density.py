import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb

# Initialize W&B
wandb.init(project="hops_pipeline", entity="marina-dominguez", name="density_preprocess")

# Load the combined dataset
with open('/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/combined_annotated_august_prostates.json', 'r') as f:
    data = json.load(f)

# Load the annotated_slices_per_patient
with open('/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/annotated_slices_per_patient.json', 'r') as f:
    annotated_slices_per_patient = json.load(f)

initial_len = len(data)
print(f"Initial length: {initial_len}")

def filter_patients_by_min_annotations(data, min_annotations=4):
    """
    Filters out patients who have fewer than the minimum number of annotations.

    Parameters:
    - data: List of patient data
    - min_annotations: Minimum number of annotations required for a patient to be kept.

    Returns:
    - filtered_data: List of patients who meet the annotation criterion.
    """
    filtered_data = []

    for patient in data:
        patient_id = patient['Folder_id']
        num_annotations = len(annotated_slices_per_patient.get(patient_id, []))

        if num_annotations >= min_annotations:
            # If the patient has at least the minimum number of annotations, keep the patient
            filtered_data.append(patient)
        else:
            print(f"Excluding patient {patient_id} with {num_annotations} annotations.")
    
    final_len = len(filtered_data)
    print(f"Final length: {final_len}")
    print(f"Removed {initial_len - final_len} patients with fewer than {min_annotations} annotations.")

    return filtered_data

def filter_patients_by_max_gap(patients_data, max_allowed_gap=4):
    """
    Filters out patients who have a maximum gap larger than the allowed threshold.

    Parameters:
    - patients_data: Dictionary of patient data (grouped by patient ID)
    - max_allowed_gap: Maximum allowed gap between annotated slices for a patient to be kept.

    Returns:
    - filtered_patients_data: Dictionary with patients who meet the gap criterion.
    """
    print("Initial dictionary size:", len(patients_data))

    filtered_patients_data = {}

    for patient_id, slices in patients_data.items():
        # Sort slices by their slice number
        slices = sorted(slices, key=lambda x: int(x['slice'].split('_')[-1].replace('.dcm', '').split('/')[-1]))
        print(f"Processing patient {patient_id} with {len(slices)} slices...")

        # Get slice numbers for annotated slices
        annotated_slices = [int(slice_data['slice'].split('_')[-1].replace('.dcm', '').split('/')[-1]) for slice_data in slices if any(pixel > 0 for pixel in slice_data['label'])]
        print(f"Annotated slices: {annotated_slices}")

        if len(annotated_slices) < 2: # not necessary, since we already filtered out patients with less than 4 annotations
            # but just as a sanity check, we can keep it
            # making sure that we never enter this condition
            print(f"Excluding patient {patient_id} with fewer than 2 annotated slices.")
            continue  # Skip if there are fewer than two annotated slices

        # Calculate gaps between consecutive annotated slices
        gaps = np.diff(annotated_slices) - 1  # Subtract 1 to get the number of unannotated slices between two annotations
        print(f"Gaps: {gaps}")

        max_gap = np.max(gaps) if len(gaps) > 0 else 0
        print(f"Max gap: {max_gap}")

        if max_gap <= max_allowed_gap:
            # If the max gap is less than or equal to the allowed threshold, keep the patient
            filtered_patients_data[patient_id] = slices
        else:
            print(f"Excluding patient {patient_id} with max gap of {max_gap} slices.")
        
    print("Final dictionary size:", len(filtered_patients_data))

    return filtered_patients_data

def interpolate_annotations(slices):
    """
    Fills gaps in the annotated slices using linear interpolation, but retains all slices (annotated and non-annotated).

    Parameters:
    - slices: List of slice data sorted by slice number, including both annotated and non-annotated slices.

    Returns:
    - slices: List of slices with interpolated annotations for missing slices and all original slices.
    """
    # Get only annotated slices with their slice numbers
    annotated_slices = [slice_data for slice_data in slices if any(pixel > 0 for pixel in slice_data['label'])]
    slice_numbers = [int(slice_data['slice'].split('_')[-1].replace('.dcm', '').split('/')[-1]) for slice_data in slices]

    if len(annotated_slices) < 2:
        # Not enough annotated slices to interpolate, return all slices as is
        return slices

    # Create a list to hold the final data including interpolated slices
    final_slices = []
    
    # Iterate over all slices and perform interpolation only for annotations
    for i in range(len(annotated_slices) - 1):
        current_slice = annotated_slices[i]
        next_slice = annotated_slices[i + 1]
        current_index = slice_numbers.index(int(current_slice['slice'].split('_')[-1].replace('.dcm', '').split('/')[-1]))
        next_index = slice_numbers.index(int(next_slice['slice'].split('_')[-1].replace('.dcm', '').split('/')[-1]))

        # Add the current slice to the final list
        final_slices.append(current_slice)

        # Check if there are gaps between the current and next annotated slice
        gap_size = next_index - current_index - 1

        if gap_size > 0:
            print(f"Interpolating {gap_size} slice(s) between slice {current_slice['slice']} and {next_slice['slice']}")
            # Linear interpolation of label values between the two annotated slices
            for j in range(1, gap_size + 1):
                interpolated_slice = slices[current_index + j].copy()
                # Perform linear interpolation only for the labels
                current_label = np.array(current_slice['label'])
                next_label = np.array(next_slice['label'])
                interpolated_slice['label'] = ((current_label * (gap_size - j + 1) + next_label * j) / (gap_size + 1)).tolist()
                final_slices.append(interpolated_slice)
    
    # Add the last annotated slice
    final_slices.append(annotated_slices[-1])

    # Add all remaining non-annotated slices (those that were not processed)
    # but they can not be in between annotated slices, those should be interpolated, not black slices
    for slice_data in slices:
        # if they are before the first annotated slice, we keep them
        if slice_data not in final_slices and \
              slice_numbers.index(int(slice_data['slice'].split('_')[-1].replace('.dcm', '').split('/')[-1])) \
                < slice_numbers.index(int(annotated_slices[0]['slice'].split('_')[-1].replace('.dcm', '').split('/')[-1])):
            final_slices.append(slice_data)
        # if they are after the last annotated slice, we keep them
        if slice_data not in final_slices and \
            slice_numbers.index(int(slice_data['slice'].split('_')[-1].replace('.dcm', '').split('/')[-1])) \
                > slice_numbers.index(int(annotated_slices[-1]['slice'].split('_')[-1].replace('.dcm', '').split('/')[-1])):
            final_slices.append(slice_data)
        else: # if they are in between annotated slices, we do not keep them
            pass

    return sorted(final_slices, key=lambda x: int(x['slice'].split('_')[-1].replace('.dcm', '').split('/')[-1]))

"""def plot_comparison_between_og_patient_and_interpolated_patient(og_patient, interpolated_patient):
    # Get the slice numbers and labels for the original patient
    og_slice_numbers = [int(slice_data['slice'].split('_')[-1].replace('.dcm', '').split('/')[-1]) for slice_data in og_patient]
    og_labels = [slice_data['label'] for slice_data in og_patient]

    # Get the slice numbers and labels for the interpolated patient
    interpolated_slice_numbers = [int(slice_data['slice'].split('_')[-1].replace('.dcm', '').split('/')[-1]) for slice_data in interpolated_patient]
    interpolated_labels = [slice_data['label'] for slice_data in interpolated_patient]

    # Plot the original patient
    plt.figure(figsize=(12, 6))
    plt.plot(og_slice_numbers, og_labels, 'o-', label='Original Patient')
    plt.plot(interpolated_slice_numbers, interpolated_labels, 'x-', label='Interpolated Patient')
    plt.xlabel('Slice Number')
    plt.ylabel('Label Value')
    plt.title('Comparison Between Original and Interpolated Patient')
    plt.legend()
    plt.grid()

    # Save plot as an image and log to wandb
    plt_image = wandb.Image(plt)  # Convert the plot to an image
    wandb.log({"Comparison Between Original and Interpolated Patient": plt_image})

    plt.close()

def five_five_remove_notlabelled_slices(data):
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
    """

# Convert the dictionary (interpolated_patients_data) back into a flat list
def flatten_interpolated_data(interpolated_patients_data):
    """
    Converts the dictionary of interpolated patient data back into a flat list of slices.

    Parameters:
    - interpolated_patients_data: Dictionary where each key is a patient_id and the value is a list of slices.

    Returns:
    - flat_list: A flat list of all the slices across patients.
    """
    flat_list = []
    for patient_id, slices in interpolated_patients_data.items():
        for slice_data in slices:
            # Make sure to include the 'Folder_id' for each slice
            slice_data['Folder_id'] = patient_id
            flat_list.append(slice_data)
    return flat_list


print("Applying first filtering based on number of annotations...")
removed_not_enough_annotated_data = filter_patients_by_min_annotations(data, min_annotations=4)

print("----------------------------------------")
print("Applying second filtering based on maximum gap between annotated slices...")
patients_data = {}
for entry in removed_not_enough_annotated_data:
    patient_id = entry['Folder_id']
    if patient_id not in patients_data:
        patients_data[patient_id] = []
    patients_data[patient_id].append(entry)

filtered_patients_data = filter_patients_by_max_gap(patients_data, max_allowed_gap=4)

print("----------------------------------------")
print("Interpolating annotations for patients with gaps...")
interpolated_patients_data = {}
for patient_id, slices in filtered_patients_data.items():
    print(f"Interpolating annotations for patient {patient_id}...")
    interpolated_slices = interpolate_annotations(slices)
    interpolated_patients_data[patient_id] = interpolated_slices

print("----------------------------------------")
print("Saving the filtered data in the specified output path as a JSON file...")
# Flatten the interpolated data back into a list
flat_interpolated_data = flatten_interpolated_data(interpolated_patients_data)

# Save the flattened data
output_path = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/preprocessing/density_preprocess/high_density_interpolated_annotated_prostates.json'
print(f"Saving the high density interpolated data to {output_path}...")
with open(output_path, 'w') as f:
    json.dump(flat_interpolated_data, f)

print("Saved and completed!")