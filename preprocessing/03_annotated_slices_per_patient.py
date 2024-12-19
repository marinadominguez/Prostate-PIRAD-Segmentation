"""

This script takes as input the combined annotation JSON file produced by the previous steps and performs several analytic tasks, such as:
Checking the existence of the combined annotation file.
Counting how many slices are annotated per patient.
Creating a histogram illustrating the distribution of the number of annotated slices per patient.
Counting and listing the number of unique patients and identifying patients with multiple entries.
Saving a JSON file summarizing which slices are annotated for each patient, necessary for the next interpolation step of the pipeline.

"""

import json
import ast
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import os

combined_annotated_prostates = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/combined_annotated_PIRAD3.json'

def check_file_exists(filepath):
    """
    Check if the specified file exists.
    """
    if not os.path.exists(filepath):
        print('File does not exist')
        exit()
    print('File exists')

def load_json(filepath):
    """
    Load JSON data from the specified file.
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def count_labeled_slices(data):
    """
    Count the number of labeled slices per patient and return as a DataFrame.
    """
    dict_labeledslices_patient = {}
    for entry in data:
        if entry['label'] != 0:
            count = sum(1 for label in entry['label'] if label != 0)
            dict_labeledslices_patient[entry['Folder_id']] = count
    dataframe = pd.DataFrame.from_dict(dict_labeledslices_patient, orient='index', columns=['PIRAD3']).reset_index().rename(columns={'index': 'Folder_id'})
    return dataframe

def plot_histogram(df):
    """
    Plot a histogram of the number of labeled slices per patient.
    """
    plt.hist(df['PIRAD3'], bins=20, edgecolor='black')
    plt.title('Distribución del número de slices etiquetadas por próstata')
    plt.xlabel('Número de slices etiquetadas')
    plt.ylabel('Número de pacientes')
    plt.show()
    plt.savefig('/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/PIRAD3/histogram_labeled_slices_december.png')
    plt.close()

def unique_patient_count(data):
    """
    Print the number of unique patients and any with multiple entries.
    """
    folder_id_counts = Counter(item['Folder_id'] for item in data)
    unique_patients = len(folder_id_counts)
    multiple_entries = {folder_id: count for folder_id, count in folder_id_counts.items() if count > 1}
    print(f"Number of unique patients: {unique_patients}")
    print(f"Patients with multiple entries: {len(multiple_entries)}")
    for folder_id, count in list(multiple_entries.items())[:5]:
        print(f"Patient {folder_id} has {count} entries.")

def count_annotated_slices(data):
    """
    Count and store annotated slices per patient.
    """
    annotated_slices = defaultdict(list)
    count_annotated = 0
    current_patient_id = None
    slice_index = 0

    for entry in data:
        folder_id = entry['Folder_id']
        if folder_id != current_patient_id:
            current_patient_id = folder_id
            slice_index = 0  # Reset slice index for the new patient
        if any(pixel > 0 for pixel in entry['label']):
            count_annotated += 1
            annotated_slices[folder_id].append(slice_index)
        slice_index += 1

    print(f"Total annotated slices: {count_annotated}")
    for patient_id, slices in annotated_slices.items():
        print(f"Patient {patient_id}: {len(slices)} annotated slices at indices {slices}")
    return annotated_slices

def save_to_json(data, filepath):
    """
    Save the data to a JSON file.
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    check_file_exists(combined_annotated_prostates)
    data = load_json(combined_annotated_prostates)
    
    # Count labeled slices
    df = count_labeled_slices(data)
    print(df.head())
    
    # Plot histogram
    plot_histogram(df)
    
    # Unique patient count
    unique_patient_count(data)
    
    # Count annotated slices
    annotated_slices = count_annotated_slices(data)
    
    # Save results to JSON
    save_to_json(annotated_slices, '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/PIRAD3/annotated_slices_per_patient_december.json')

if __name__ == "__main__":
    main()
