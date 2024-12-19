# 3d-UNet-PIRAD-Segmentation

## Overview
This project implements a 3D U-Net-based pipeline for prostate segmentation using PIRADS data. The goal is to improve prostate imaging and diagnosis through automated segmentation.

## Features
- 3D U-Net architecture tailored for medical imaging.
- Support for PIRADS data preprocessing and augmentation.
- End-to-end segmentation pipeline.
- Customizable training and evaluation scripts.

## Requirements
To run this project, you need:
- Python 3.8 or higher
- TensorFlow or PyTorch (Specify framework)
- Required Python libraries (see `requirements.txt`)

## Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/3d-UNet-PIRAD-Segmentation.git
   ```

2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

3. Add your data to the data/ directory (not included in this repository for privacy).

## Preprocessing Pipeline for Prostate MRI Data

This directory provides a comprehensive preprocessing pipeline for transforming raw DICOM images and corresponding segmentation annotations into a finalized dataset ready for deep learning model training. The pipeline includes scripts for filtering patients, extracting annotations, handling multiple annotation strategies (single-label, multi-label, or merged-label), interpolating missing slices, and optionally cropping images around the prostate region and lesions.

### Overview

### Goals of the Preprocessing Pipeline
- **Filter & Extract:** Identify relevant patients and remove those without any annotations.
- **Preprocess Annotations:** Convert raw annotations into desired formats:
  - **All Annotations as-is:** Keep prostate and lesion annotations (PIRADS-3, PIRADS-4, PIRADS-5) separate.
  - **Single Annotation Type:** Extract only one specific annotation category (e.g., PIRADS-3 only).
  - **Multiple Annotations as One:** Merge several annotation categories into a binary segmentation (e.g., any PIRADS≥3 lesion as “cancer”).
- **Annotate Slices per Patient:** Analyze the resulting dataset to understand the distribution of annotations and identify annotated slices per patient.
- **Interpolate & Finalize:** Interpolate annotations between slices to create smoother volumes, and filter out patients with insufficient annotations or large annotation gaps.
- **(Optional) Crop Regions:** Once the dataset is finalized, optionally crop images around the prostate region and corresponding lesions to reduce input size and potentially focus model training on the region of interest.

### Final Output
The final result of this pipeline is a JSON file containing patient IDs and their processed volumes, along with annotation masks. You can optionally produce a cropped dataset based on the prostate’s bounding box and save the cropped images and labels as separate PNG files, along with a JSON file describing the new cropped dataset.

## When to Run the Preprocessing Steps

You have two main options:

1. **Full Preprocessing from Scratch**:  
   If you have **raw DICOM data and the initial JSON annotation files**, you can run the entire pipeline to produce your final dataset. This approach ensures full reproducibility and flexibility in how you handle annotations.  
   
   Steps:
   - Run `01_filter_and_extract_annotations.py` to filter patients and produce an initial annotated dataset.
   - Pick one of the preprocessing approaches in step 2 (depending on how you want to handle annotations):
     - `02_preprocess_all_annotations.py`  
     - `02_preprocess_one_annotation.py`  
     - `02_preprocess_multiple_annotations_as_one.py`  
   - Run `03_annotated_slices_per_patient.py` to analyze and summarize annotated slices per patient.
   - Run `04_interpolate_and_finalize_annotations.py` to finalize and interpolate the dataset.
   - **Optionally**, run `05_preprocess_cropping.py` from the `cropping_imgs_prostates&lessions` folder to crop images around the prostate region. This step uses the final JSON file produced by step 4 and a trained model to identify the prostate region and create smaller, focused volumes.

   After these steps, you will have a final JSON file (and optionally cropped images and labels) ready for training.

2. **Skipping Directly to Training with Provided Data**:  
   If you **already have the final JSON file** (or the cropped dataset) from a previous run, you can skip all preprocessing steps. Just place your final JSON and associated data where your training code expects them, and proceed directly to model training using `dataset_hosp.py` as your dataloader.

## Folder Structure and Key Scripts

- **`01_filter_and_extract_annotations.py`**  
  Filters out patients with no annotations and creates a summary Excel and JSON file of valid patients.

- **`02_preprocess_all_annotations.py`**, **`02_preprocess_one_annotation.py`**, **`02_preprocess_multiple_annotations_as_one.py`**  
  Scripts offering different preprocessing strategies:
  - Keep all annotations separate (`02_preprocess_all_annotations.py`)
  - Extract a single annotation type (`02_preprocess_one_annotation.py`)
  - Merge multiple annotation types into one binary label (`02_preprocess_multiple_annotations_as_one.py`)

- **`03_annotated_slices_per_patient.py`**  
  Analyzes the processed dataset, counting how many annotated slices each patient has, and producing a JSON mapping patient IDs to their annotated slices.

- **`04_interpolate_and_finalize_annotations.py`**  
  Filters patients based on minimum number of annotations and maximum allowed gap between annotated slices. It also interpolates annotations to create smoother volumes, producing a final JSON ready for the training dataloader.

- **`cropping_imgs_prostates&lessions/05_preprocess_cropping.py`** (Optional)  
  Uses the final dataset and a trained prostate model to identify the prostate region. It then crops images and annotations around the prostate and lesion areas, saving the cropped slices and a new JSON file describing these cropped volumes.

- **`dataset_hosp.py`**  
  A PyTorch dataloader that reads the final (or cropped) dataset JSON file and converts it into trainable datasets. This script is used during the model training stage.

## Requirements and Configuration

- Ensure Python 3.8+ and required dependencies are installed.
- Configure `HOSP_pipeline/config/config.py` with:
  - `DATASET_PATH` pointing to the final JSON file after preprocessing (or the cropped dataset if using the optional step).
  - Splitting parameters (`TRAIN_VAL_TEST_SPLIT`, `SPLIT_SEED`) and other training hyperparameters.
- If preprocessing from scratch:
  - Ensure the raw DICOM data and initial JSON annotation files are accessible.
  - Adjust paths within each script or use command-line arguments or configuration files as needed.

## Training the Model

Once you have successfully generated the final dataset via the preprocessing pipeline (and optionally performed cropping), you can proceed to train your model using the provided training script. This script loads the final (and optionally cropped) dataset, sets up data augmentation transforms, and trains a 3D UNet model for prostate segmentation with your chosen loss functions and hyperparameters.

**Key Points:**

- **Input Data**: The training script uses the final JSON file produced by the preprocessing pipeline to load the dataset via `dataset_hosp.py`.
- **Transforms**: Before training, data augmentation or normalization transforms can be applied to the training and validation sets.
- **Model Architecture**: A 3D UNet (`HOSP_pipeline/models/unet3d.py`) is initialized, supporting multiple input channels and output classes.
- **Loss Functions**: Various loss functions (e.g., focal loss, dice-based losses, etc.) are supported. The script uses `CustomFocalForegroundDiceLoss` by default, defined in `HOSP_pipeline/training/my_focal_loss.py`.
- **CUDA Support**: If CUDA is available and `TRAIN_CUDA` is set, training will occur on the GPU. Otherwise, the script falls back to CPU mode.
- **Logging & Visualization**: The script logs training and validation losses, prints foreground coverage statistics, and can optionally visualize intermediate outputs to ensure model predictions are as expected.
- **Checkpoints & Early Stopping**: Model weights are saved periodically (at defined intervals or when a new best validation loss is found). Early stopping criteria can be configured to halt training if no improvement is observed after a certain number of epochs.

### Steps to Run the Training

1. **Ensure Preprocessing Completed**:  
   Make sure you have the final JSON dataset file ready. 

2. **Update `config.py`**:  
   Make sure that `HOSP_pipeline/config/config.py` points to the correct `DATASET_PATH`, batch sizes, number of epochs, and other parameters matching your hardware and training preferences.

3. **Run the Training Script**:  
   From the project’s main directory:
   ```bash
   python trainandval_wtrans_sigmoid.py
   ```

   This will:
   - Load the dataset via `get_train_val_test_Dataloaders()` using the paths and parameters defined in `config.py`.
   - Initialize the model, optimizer, and loss function.
   - Run the training loop for the specified number of epochs, saving checkpoints and printing out performance metrics.

4. **Monitor Results**:  
   Check the console output for training and validation losses, dice scores, and foreground coverage metrics.  
   
   The script saves model checkpoints at `SAVE_CHECKPOINT_PATH` (defined in `config.py`) and can save diagnostic images under `output/training_outputs` and `output/validation_outputs` directories for visualization.

5. **Early Stopping and Best Models**:  
   If early stopping is enabled and triggered, training will stop early if no improvement is observed. The best validation model will remain saved as a checkpoint. Use this model checkpoint to evaluate on test sets, refine further, or deploy in applications.



## Usage

- Train the model:
  ```bash
  python train.py --config configs/train_config.yml
  ```
  
- Evaluate the model:
  ```bash
  python evaluate.py --checkpoint checkpoints/model.pth
  ```
  
- Run segmentation: 
  ```bash
  python segment.py --input test_images/ --output results/
  ```
  
## Acknowledgements
This project was inspired by the work of [source/reference].
Credits to the open-source libraries used in this project.

