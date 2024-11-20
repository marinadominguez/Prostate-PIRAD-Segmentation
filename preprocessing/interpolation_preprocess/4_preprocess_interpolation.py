import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from scipy.ndimage import distance_transform_edt

# Initialize W&B
wandb.init(project="hops_pipeline", entity="marina-dominguez", name="density_preprocess")

# Load the combined dataset
with open('/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/combined_annotated_october_prostates.json', 'r') as f:
    data = json.load(f)

# Load the annotated_slices_per_patient (which is )
with open('/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/annotated_slices_per_patient_october.json', 'r') as f:
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
            #print(f"Current label shape: {current_label.shape}") # Current label shape: (262144,)
            #print(f"Current label: {current_label}")  # Current label: [0. 0. 0. ... 0. 0. 0.]
            #print(f"Current label type: {type(current_label)}")  # Current label type: <class 'numpy.ndarray'>
            #print(f"Current label shape (if applicable): {current_label.shape if isinstance(current_label, np.ndarray) else 'Not an array'}") # Current label shape (if applicable): (262144,)
            next_label = np.array(next_slice['label'])
            # print(f"Next label shape: {next_label.shape}") # Next label shape: (262144,)

            # Compute the union and intersection of the two label arrays
            union = np.maximum(current_label, next_label)  # 1 where either slice has a label
            intersection = np.minimum(current_label, next_label)  # 1 where both slices have labels
            union_minus_intersection = union - intersection
            
            # Compute the distance transform (distance of each pixel to the closest 1 in the intersection)
            # The distance map will be 0 for pixels in the intersection and increase for pixels in union_minus_intersection
            distance_map = distance_transform_edt(1 - intersection)  # Distance from intersection (1s) to union_minus_intersection

            # Sort the distances to make sure we're selecting pixels in a progressive manner
            sorted_distances = np.sort(distance_map[union_minus_intersection > 0])
            #print(f"Sorted distances: {sorted_distances}")
            num_union_pixels = len(sorted_distances)
            #print(f"Number of union pixels: {num_union_pixels}")

            for j in range(1, gap_size + 1):
                # Calculate weight for this interpolated slice
                weight = j / (gap_size + 1)
                #print(f"Weight: {weight}")
                # this weight is going to determine how many closed pixels to the intersection we are going to accept as the new interpolated label
                # the new interpolated labels will be the union of the intersection and the pixels in the union_minus_intersection that are closer to the intersection in proportion to the weight
                # the closer to the intersection, the more likely to be part of the new interpolated label
                # the further from the intersection, the less likely to be part of the new interpolated label
                # the weight will determine the proportion of the pixels in the union_minus_intersection that are closer to the intersection that will be part of the new interpolated label 

                # Determine the threshold distance based on the weight (grow progressively)
                num_pixels_to_select = int(weight * num_union_pixels)
                #print(f"Number of pixels to select: {num_pixels_to_select}")
                if num_pixels_to_select > 0:
                    threshold = sorted_distances[num_pixels_to_select - 1]  # Find the distance threshold
                    #print(f"Threshold: {threshold}")
                    # Select pixels that are within the distance threshold
                    selected_pixels = (distance_map <= threshold).astype(int) * union_minus_intersection
                    # The interpolated mask will be the intersection plus a proportion of the difference
                    interpolated_label = intersection + selected_pixels
                else:
                    # If there are no pixels to select, just use the intersection
                    interpolated_label = intersection
                
                # Threshold the distance map to grow the region progressively
                #threshold = np.percentile(distance_map[union_minus_intersection > 0], weight * 100)
                #print(f"Threshold: {threshold}")

                # Select pixels that are within the distance threshold
                #selected_pixels = (distance_map <= threshold).astype(int) * union_minus_intersection

                # The interpolated mask will be the intersection plus a proportion of the difference
                #interpolated_label = intersection + selected_pixels
                
                """
                if interpolated_label.sum() > 0: 
                    # resize (262144,) to (512, 512)
                    if np.isscalar(current_label):  # If it's a single scalar value (0 or 1)
                        print(f"Current label is a scalar")
                        scalar_label = current_label  # This will be 0 or 1
                        reshaped_current_label = np.full((512, 512), scalar_label)
                        print(reshaped_current_label.shape)  # This will print (512, 512)
                    else:
                        # Perform your normal reshaping/interpolation if it's a full image
                        reshaped_current_label = current_label.reshape(512, 512)

                    if np.isscalar(interpolated_label):  # If it's a single scalar value (0 or 1)
                        print(f"Interpolated label is a scalar")
                        scalar_label = interpolated_label  # This will be 0 or 1
                        reshaped_interpolated_label = np.full((512, 512), scalar_label)
                        print(reshaped_interpolated_label.shape)  # This will print (512, 512)
                    else:
                        # Perform your normal reshaping/interpolation if it's a full image
                        reshaped_interpolated_label = interpolated_label.reshape(512, 512)
                    
                    if np.isscalar(next_label):  # If it's a single scalar value (0 or 1)
                        print(f"Interpolated label is a scalar")
                        scalar_label = next_label  # This will be 0 or 1
                        reshaped_next_label = np.full((512, 512), scalar_label)
                        print(reshaped_next_label.shape)  # This will print (512, 512)
                    else:
                        # Perform your normal reshaping/interpolation if it's a full image
                        reshaped_next_label = next_label.reshape(512, 512)
                    
                    # make a 1x3 subplot
                    plt.subplot(1, 3, 1) # 1 row, 3 columns, 1st subplot
                    plt.imshow(reshaped_current_label, cmap='gray')
                    plt.title("Current label")
                    plt.axis('off')
                    plt.subplot(1, 3, 2) # 1 row, 3 columns, 2nd subplot
                    plt.imshow(reshaped_interpolated_label, cmap='gray')
                    plt.title("Interpolated label")
                    plt.axis('off')
                    plt.subplot(1, 3, 3) # 1 row, 3 columns, 3rd subplot
                    plt.imshow(reshaped_next_label, cmap='gray')
                    plt.title("Next label")
                    plt.axis('off')
                    plt.show()
                    wandb.log({"Interpolated label": [wandb.Image(plt, caption="First Interpolated label")]})
                    plt.close()
                    """

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
output_path = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/preprocessing/interpolation_preprocess/interpolated_annotated_prostates_october.json'
print(f"Saving the high density interpolated data to {output_path}...")
with open(output_path, 'w') as f:
    json.dump(flat_interpolated_data, f)

print("Saved and completed!")
wandb.finish()