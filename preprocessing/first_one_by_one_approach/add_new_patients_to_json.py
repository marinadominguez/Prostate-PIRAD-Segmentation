import os
import json

def load_json_file(file_path):
    """
    Load a JSON file from the specified path.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The contents of the JSON file.
    """
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        return json.load(f)

def add_patient_data_to_combined_file(new_json_file_path, patient_id, combined_file_path):
    """
    Add a new patient's data to the combined JSON file.

    Args:
        new_json_file_path (str): The path to the new patient's JSON file.
        patient_id (str): The ID of the new patient.
        combined_file_path (str): The path to the combined JSON file.
    """
    # Load the existing combined data
    combined_data = load_json_file(combined_file_path)

    # Load the new patient's data
    new_patient_data = load_json_file(new_json_file_path)

    # Append new patient's data to the combined data
    for entry in new_patient_data:
        combined_data.append({
            "slice": entry['slice'],
            "label": entry['label'],
            "Patient_id": patient_id
        })
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(combined_file_path), exist_ok=True)
    
    # Write the updated combined data to the JSON file
    with open(combined_file_path, 'w') as f:
        json.dump(combined_data, f, indent=4)
    print(f"Updated combined JSON file saved to {combined_file_path}")

# Example usage
new_json_file_path = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/prostate_pat_25.json'
patient_id = '25'
combined_file_path = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/combined_annotated_prostates.json'

add_patient_data_to_combined_file(new_json_file_path, patient_id, combined_file_path)
