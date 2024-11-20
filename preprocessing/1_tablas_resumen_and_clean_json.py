import json
import os
import ast
import pandas as pd

# Define the file names
input_file = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/24_10_2024/annotations_24_Oct.json'
#'/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/23_09_2024/23_sept.json'
#'/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/29_08_2024/data_august_2024.json'
dicom_file = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/dicoms.json' # DON'T CHANGE IT, this one is the dicom with all the information, 
output_json_file = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/24_10_2024/all_annotated_patients_october_2024.json'
output_excel_file = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/24_10_2024/all_annotated_patients_october_2024.xlsx'

# Check if the input file exists
if not os.path.exists(input_file) or not os.path.exists(dicom_file):
    print('One or both of the input files do not exist')
    exit()

# Read the input file
with open(input_file, 'r') as f:
    data = json.load(f)

# Read the dicom mapping file
with open(dicom_file, 'r') as f:
    dicom_data = json.load(f)

# Extract the "data" array from dicoms.json, which contains the mappings
dicom_records = dicom_data[2]['data']

# Create a mapping from dicom_file_id to dicom folder name
dicom_mapping = {item['id']: item['dicom'] for item in dicom_records}

# Assuming 'data' is a list and the relevant data is in the third element
dataset = data[2]['data']

# Create a list to store rows for the Excel file
excel_data = []

# Filter out patients where 'labelmap1' does not contain any '0'
filtered_dataset = []
for patient in dataset:
    labelmap1 = ast.literal_eval(patient['labelmap1'])
    labelmap2 = ast.literal_eval(patient['labelmap2'])
    labelmap3 = ast.literal_eval(patient['labelmap3'])
    labelmap4 = ast.literal_eval(patient['labelmap4'])
    if len(labelmap1) == 0 and len(labelmap2) == 0 and len(labelmap3) == 0 and len(labelmap4) == 0:
        continue
    else:
        filtered_dataset.append(patient)

        # Retrieve the dicom folder name from the mapping
        dicom_folder = dicom_mapping.get(str(patient['dicom_file_id']), 'Unknown')
        print(f"Patient {patient['dicom_file_id']} - {dicom_folder}")

        # Add the relevant information to the excel_data list
        excel_data.append({
            'FOLDER (dicom patient)': dicom_folder,
            'PATIENT (dicom_file_id)': patient['dicom_file_id'],
            'RADIOLOGIST (user_id)': patient['user_id'],
            'FOLDER (dicom labeled)': patient['folder'],
            'zona': patient['zona'],
            'PROSTATE(labelmap1)': len(labelmap1), 
            'PIRAD-3 (labelmap2)': len(labelmap2),
            'PIRAD-4 (labelmap3)': len(labelmap3),
            'PIRAD-5 (labelmap4)': len(labelmap4),
            'contraste': patient['contraste']
        })

# Save the filtered dataset to a new JSON file
data[2]['data'] = filtered_dataset
with open(output_json_file, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Filtered dataset saved to {output_json_file}")

# Convert the excel_data list to a DataFrame
df = pd.DataFrame(excel_data)

# Save the DataFrame to an Excel file
df.to_excel(output_excel_file, index=False)

print(f"Excel file saved to {output_excel_file}")