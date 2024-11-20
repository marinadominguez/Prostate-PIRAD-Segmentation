############################# ------- DATASET CONFIGURATIONS ------- ######################################
# 1. DATASET_PATH: Path to the dataset
DATASET_PATH = '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/preprocessing/interpolation_preprocess/interpolated_annotated_prostates_october.json'
# '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/preprocessing/soft_preprocess/soft_filtered_annotated_prostates.json'
# 
# '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/preprocessing/density_preprocess/high_density_interpolated_annotated_prostates.json'
# '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/dataset/combined_annotated_august_prostates.json'
# "/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/preprocessing/hard_preprocess/filtered_only_annotates_prostates.json"
# 2. TASK_ID: Task ID of the dataset "05": "Prostate" 
TASK_ID = 5
# 3. IN_CHANNELS: Number of input channels
IN_CHANNELS = 1
# 4. NUM_CLASSES: Number of classes to segment #4 #5 #PICAI 2,3,4,5 #number of regions you want to segment (PZ, TZ) plus one for the background.
NUM_CLASSES = 1
# 5. BACKGROUND_AS_CLASS: If True, the model treats background as a class
BACKGROUND_AS_CLASS = True
# 6. BASE_PATH: Base path of the dataset
BASE_PATH = '/ssds/DATA/Prostate/MRI/FILTERED_HOSP'
# '/ssds/DATA/Prostate/MRI/IMGS'
# 7. MODE: Mode of the pipeline
MODE = 'train'
# LOSS PARAMETERS
GAMMA = 2
ALPHA = 0.7 #0.25
#DICE_WEIGHT = 0.8
#FOREGROUND_WEIGHT = 0.8
# 8. CHECKPOINT_PATH: Path to the checkpoint
CHECKPOINT_PATH = "/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/correct_interpolation/trained_from_scratch_customfocalloss_alpha0.7/best_model_epoch93.pt"
# '/home/marina/Escritorio/Task05_Prostate/3D-UNet/checkpoints_first_seg_prostate_unet/best_model_epoch95.pt'
# '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/correct_interpolation/finetuned/fine_tuned_model_interpolated_preprocessed.pt'
# '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/5-5-onlyannotated_preprocess/trained_from_scratch_alpha0.7_dataOct_5-5/best_model_epoch74.pt'
# '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/5-5-onlyannotated_preprocess/fine_tuned_alpha0.7_5-5/fine_tuned_model_5-5_alpha0.7.pt'
# '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/5-5-onlyannotated_preprocess/fine_tuned_alpha0.7_dataOct/fine_tuned_model_soft_filtering.pt'

# 

# '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/correct_interpolation/trained_from_scratch_customfocalDiceLoss_alpha0.25/best_model_epoch90.pt'
# '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/correct_interpolation/trained_from_scratch_customfocalloss_alpha0.25/best_model_epoch90.pt'
# '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/correct_interpolation/trained_from_scratch_customfocalDiceLoss_alpha0.7/best_model_epoch90.pt'
# '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/correct_interpolation/finetuned/fine_tuned_model_interpolated_preprocessed.pt'
# '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/correct_interpolation/trained_from_scratch_FocalDice_alpha0.25_diceweight0.8_foreground0.8/best_model_epoch99.pt'

#"/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/correct_interpolation/trained_from_scratch_customfocalloss_alpha0.25/best_model_epoch90.pt"
#'/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/interpolation_preprocess/segmodels_pytorch_3d_trials/correct_interpolation_WeightedDice/best_model_epoch90.pt'
# '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/interpolation_preprocess/segmodels_pytorch_3d_trials/correct_interpolation_diceLoss/best_model_epoch49.pt'
#'/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/interpolation_preprocess/segmodels_pytorch_3d_trials/correct_interpolation/best_model_epoch40.pt' 
#
# '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/interpolation_preprocess/trained_from_scratch/best_model_epoch99.pt'
# "/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/interpolation_preprocess/fine_tuned/fine_tuned_model_interpolated_preprocessed.pt"
# '/home/marina/Escritorio/Task05_Prostate/3D-UNet/checkpoints_first_seg_prostate_unet/best_model_epoch95.pt'
# "/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/trained_from_scratch/best_model_epoch99.pt"
# "/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/finetuned_models/fine_tuned_model_soft_filtering.pt"
#
# "/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/finetuned_models/fine_tuned_model_hard_filtering_only_annotated.pt"
# 
# "/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/finetuned_models/preprocessed_fine_tuned_model.pt" 
SAVE_CHECKPOINT_PATH = "/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/5-5-onlyannotated_preprocess/trained_from_scratch_alpha0.7_dataOct_5-5"
# "/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/correct_interpolation/finetuned"
# '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/correct_interpolation/trained_from_scratch'
# '/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/interpolation_preprocess/segmodels_pytorch_3d_trials/correct_interpolation'
# None
# "/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/checkpoints/interpolation_preprocess/fine_tuned"
# "/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/"
# "/home/marina/Escritorio/Task05_Prostate/HOSP_pipeline/trained_from_scratch"
WANDB_TRAIN_NAME = 'Unet3d_CustomFocalLoss_100Epochs'
# 'Unet++ResNet34_WeightedDice_Interpolation_Preprocess'
#"FineTuning_IterpolatedHighDensity_CustomFocalLoss"

############################# ------- TRAINING CONFIGURATIONS ------- #######################################
TRAIN_VAL_TEST_SPLIT = [0.7, 0.15, 0.15] 
SPLIT_SEED = 42
TRAINING_EPOCH = 100
TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
TRAIN_CUDA = True
SAVE_INTERVAL = 10  # Guardar el modelo cada 'save_interval' épocas

########################### ------- TESTING CONFIGURATIONS ------- ########################################
TEST_BATCH_SIZE = 1
WANDB_TEST_NAME = "DiceLoss-Class4"
#CHECKPOINT_PATH = "checkpoints/train&val_weights/FocalLoss_100Epochs/best_model_epoch99.pt"
# BCE_WEIGHTS = [0.0001, 0.249, 0.289, 0.2119, 0.25]
# DICE_WEIGHT = 0.5
# ALPHA_TVERSKY = 0.3
# BETA_TVERSKY = 0.7
# GAMMA = 2


# Your configuration dictionary
config = {
    "dataset_path": DATASET_PATH,
    "task_id": TASK_ID,
    "in_channels": IN_CHANNELS,
    "num_classes": NUM_CLASSES,
    "background_as_class": BACKGROUND_AS_CLASS,
    "base_path": BASE_PATH,
    "mode": MODE,
    "gamma": GAMMA,
    "alpha": ALPHA,
    #"dice_weight": DICE_WEIGHT,
    #"foreground_weight": FOREGROUND_WEIGHT,
    "checkpoint_path": CHECKPOINT_PATH,
    "train_val_test_split": TRAIN_VAL_TEST_SPLIT,
    "split_seed": SPLIT_SEED,
    "training_epoch": TRAINING_EPOCH,
    "train_batch_size": TRAIN_BATCH_SIZE,
    "val_batch_size": VAL_BATCH_SIZE,
    "train_cuda": TRAIN_CUDA,
    "save_interval": SAVE_INTERVAL,
    "test_batch_size": TEST_BATCH_SIZE,
    "wandb_test_name": WANDB_TEST_NAME,
    "wandb_train_name": WANDB_TRAIN_NAME,
    "save_checkpoint_path": SAVE_CHECKPOINT_PATH
}