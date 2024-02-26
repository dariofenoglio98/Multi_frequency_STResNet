from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
import os



# VARIABLES
N_REPETITIONS = 10
USE_GPU = True
GPU_DEVICE=1 #GPU id
LEARNING_RATE = 0.0001
LEARNING_RATE2 = 0.0001
BATCH_SIZE = 256
NUM_CPUS = 2
NUM_EPOCHS = 150
NUM_THREADS = NUM_CPUS
PATIENCE = 10
TRAINING_VERBOSE = 1
CALLBACK_VERBOSE = 1
LOSS = SparseCategoricalCrossentropy()
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]

# PATHS
LOGDIR = f"Results/Results_CL_down_5Hz/"
MODELS_FOLDER = LOGDIR + "models/"
IMAGES_FOLDER = LOGDIR + "images/"

# create necessary folders
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)
if not os.path.exists(IMAGES_FOLDER):
    os.makedirs(IMAGES_FOLDER)
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

def test_on_one_dataset_mean(model, windows, labels):
    pred = model.predict(windows, batch_size=BATCH_SIZE, verbose=0)
    loss = LOSS(labels, pred).numpy().mean()
    accuracy = accuracy_score(labels, pred.argmax(axis=1))
    f1 = f1_score(labels, pred.argmax(axis=1), average='macro')
    return np.mean(loss), np.mean(accuracy), np.mean(f1)

# Define the ReduceLROnPlateau callback (not used)
def scheduler(epoch, lr):
    if epoch < 20000:
        return LEARNING_RATE
    else:
        return LEARNING_RATE/10
    
