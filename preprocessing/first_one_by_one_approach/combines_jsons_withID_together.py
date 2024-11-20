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
    with open(file_path, 'r') as f:
        return json.load(f)

def combine_json_files(patient_ids, json_file_paths, output_path):
    """
    Combine multiple JSON files into a single JSON file.

    Args:
        json_file_paths (list of str): The paths to the JSON files to combine.
        patient_ids (list of str): The patient IDs corresponding to the JSON files.
        output_path (str): The path to save the combined JSON file.
    """
    if len(json_file_paths) != len(patient_ids):
        raise ValueError("The number of JSON files must match the number of patient IDs.")
    
    combined_data = []
    for json_file_path, patient_id in zip(json_file_paths, patient_ids):
        print(f"Processing file: {json_file_path}")
        json_data = load_json_file(json_file_path)
        for entry in json_data:
            combined_data.append({
                "slice": entry['slice'],
                "label": entry['label'],
                "Patient_id": patient_id
            })
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the combined data to a JSON file
    with open(output_path, 'w') as f:
        json.dump(combined_data, f)
    print(f"Combined JSON file saved to {output_path}")

# Example usage
json_file_paths = [
    '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/prostate_pat_48.json',
    '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/prostate_pat_51.json',
    '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/prostate_pat_52.json'
]

patient_ids = [
    '48',
    '51',
    '52'
]

output_path = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/combined_annotated_prostates.json'

combine_json_files(patient_ids, json_file_paths, output_path)
