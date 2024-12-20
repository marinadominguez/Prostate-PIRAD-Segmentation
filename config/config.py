############################# ------- DATASET CONFIGURATIONS ------- ######################################
# 1. DATASET_PATH: Path to the dataset
DATASET_PATH = 'preprocessing/interpolation_preprocess/interpolated_ALL_annotated_cancer_october_minannot1_maxgap6.json'

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

# 7. MODE: Mode of the pipeline
MODE = 'train'

# LOSS PARAMETERS
GAMMA = 2
ALPHA = 0.7 #0.25
#DICE_WEIGHT = 0.8
#FOREGROUND_WEIGHT = 0.8

# 8. CHECKPOINT_PATH: Path to the checkpoint
CHECKPOINT_PATH = "checkpoints/interpolation_preprocess/trained_from_scratch_cancer_lessions_wcropping/best_model_epoch90.pt"
PROSTATE_CHECKPOINT = "checkpoints/correct_interpolation/trained_from_scratch_customfocalloss_alpha0.7/best_model_epoch93.pt"
SAVE_CHECKPOINT_PATH = 'checkpoints/interpolation_preprocess/trained_from_scratch_cancer_lessions_wcropping'

############################# ------- TRAINING CONFIGURATIONS ------- #######################################
TRAIN_VAL_TEST_SPLIT = [0.8, 0.15, 0.05] # [0.0, 0.0, 1.0]
SPLIT_SEED = 42
TRAINING_EPOCH = 100 # 50
TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
TRAIN_CUDA = True
SAVE_INTERVAL = 10  # Guardar el modelo cada 'save_interval' épocas
# We can also set the patients ids for training, validation and testing manually, here we show a symbolic example
TEST_PATIENT_IDS = [ '41e8698b51744d7', '933b4fcf62e6979']
TRAIN_PATIENTS_IDS = ['db12ced81bc2a00', '84c8b22196adbd4']
VAL_PATIENTS_IDS = ['1829ed05901f123', 'b29059098a4102f']

########################### ------- TESTING CONFIGURATIONS ------- ########################################
TEST_BATCH_SIZE = 1
SELECTED_THRESHOLD = 0.202
# CHECKPOINT_PATH = "checkpoints/train&val_weights/FocalLoss_100Epochs/best_model_epoch99.pt"
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
    "save_checkpoint_path": SAVE_CHECKPOINT_PATH, 
    "selected_threshold": SELECTED_THRESHOLD,
}