"""

This script takes the previously combined and analyzed dataset and applies final preprocessing steps prior to model training. 
It:
- Filters out patients with fewer than a specified minimum number of annotated slices.
- Filters out patients whose maximum gap between annotated slices exceeds a specified threshold.
- Uses a custom interpolation method to “fill in” unannotated slices between annotated slices, progressively adding intermediate 
    masks to create a smoother and more continuous annotation volume.
- Produces a final JSON file with interpolated annotations suitable for downstream tasks such as model training.

"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt


# Load the combined dataset
combined_dataset_path = 'dataset/combined_annotated_PIRAD3.json'
# Load the annotated_slices_per_patient 
annotated_slices_per_patient_path = 'dataset/PIRAD3/annotated_slices_per_patient_december.json'

# Load the data
with open(combined_dataset_path, 'r') as f:
    data = json.load(f)

with open(annotated_slices_per_patient_path, 'r') as f:
    annotated_slices_per_patient = json.load(f)

initial_len = len(data)
print(f"Initial length: {initial_len}")

min_annotations = 1 #4
max_allowed_gap = 6 #4

def filter_patients_by_min_annotations(data, min_annotations):
    """
    Filters out patients who have fewer than the minimum number of annotations.

    Parameters:
    - data: List of patient data
    - min_annotations: Minimum number of annotations required for a patient to be kept.

    Returns:
    - filtered_data: List of patients who meet the annotation criterion.
    """
    filtered_data = []
    testing_set = set()

    for patient in data:
        patient_id = patient['Folder_id']
        num_annotations = len(annotated_slices_per_patient.get(patient_id, []))

        if num_annotations >= min_annotations:
            # If the patient has at least the minimum number of annotations, keep the patient
            filtered_data.append(patient)
            if num_annotations == 1: 
                testing_set.add(patient_id)
        else:
            print(f"Excluding patient {patient_id} with {num_annotations} annotations.")
            if num_annotations == 1: 
                testing_set.add(patient_id)
                # since we can not use it for interpolation, and therefore training,
                # we are going to use these for testing
    
    final_len = len(filtered_data)
    print(f"Final length: {final_len}")
    print(f"Removed {initial_len - final_len} patients with fewer than {min_annotations} annotations.")
    print(f"Testing set: {testing_set}")

    return filtered_data

def filter_patients_by_max_gap(patients_data, max_allowed_gap):
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

        if len(annotated_slices) < 1: #2: # not necessary, since we already filtered out patients with less than 4 annotations
            # but just as a sanity check, making sure that we never enter this condition
            print(f"Excluding patient {patient_id} with fewer than 1 annotated slices.")
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

def iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    
    Parameters:
    - mask1: 2D numpy array (binary mask)
    - mask2: 2D numpy array (binary mask)
    
    Returns:
    - IoU: The IoU score between mask1 and mask2.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def interpolate_annotations(slices):
    """
    Fills gaps in the annotated slices using proportional interpolation, where each slice is
    an intermediate step between the previous and next annotated slice based on their union and intersection.

    Parameters:
    - slices: List of slice data sorted by slice number, including both annotated and non-annotated slices.

    Returns:
    - slices: List of slices with interpolated annotations for missing slices.
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
            # Extract labels from the current and next slices
            current_label = np.array(current_slice['label'])
            next_label = np.array(next_slice['label'])

            # Compute the union and intersection of the two label arrays
            union = np.maximum(current_label, next_label)  # 1 where either slice has a label
            intersection = np.minimum(current_label, next_label)  # 1 where both slices have labels
            union_minus_intersection = union - intersection
            
            # Compute the distance transform (distance of each pixel to the closest 1 in the intersection)
            # The distance map will be 0 for pixels in the intersection and increase for pixels in union_minus_intersection
            distance_map = distance_transform_edt(1 - intersection)  # Distance from intersection (1s) to union_minus_intersection

            # Sort the distances to make sure we're selecting pixels in a progressive manner
            sorted_distances = np.sort(distance_map[union_minus_intersection > 0])
            num_union_pixels = len(sorted_distances)

            for j in range(1, gap_size + 1):
                # Calculate weight for this interpolated slice
                weight = j / (gap_size + 1)
                
                # Determine the threshold distance based on the weight (grow progressively)
                num_pixels_to_select = int(weight * num_union_pixels)
                if num_pixels_to_select > 0:
                    threshold = sorted_distances[num_pixels_to_select - 1]  # Find the distance threshold
                    # Select pixels that are within the distance threshold
                    selected_pixels = (distance_map <= threshold).astype(int) * union_minus_intersection
                    # The interpolated mask will be the intersection plus a proportion of the difference
                    interpolated_label = intersection + selected_pixels
                else:
                    # If there are no pixels to select, just use the intersection
                    interpolated_label = intersection

                # Create a copy of the slice and assign the interpolated label
                interpolated_slice = slices[current_index + j].copy()
                interpolated_slice['label'] = interpolated_label.flatten().tolist()  # Flatten back to 1D array
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
removed_not_enough_annotated_data = filter_patients_by_min_annotations(data, min_annotations)

print("----------------------------------------")
print("Applying second filtering based on maximum gap between annotated slices...")
patients_data = {}
for entry in removed_not_enough_annotated_data:
    patient_id = entry['Folder_id']
    if patient_id not in patients_data:
        patients_data[patient_id] = []
    patients_data[patient_id].append(entry)

filtered_patients_data = filter_patients_by_max_gap(patients_data, max_allowed_gap)

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
output_path = f'preprocessing/interpolated_ALL_annotated_PIRAD3_december_minannot{min_annotations}_maxgap{max_allowed_gap}.json'
print(f"Saving the high density interpolated data to {output_path}...")
with open(output_path, 'w') as f:
    json.dump(flat_interpolated_data, f)

print("Saved and completed!")