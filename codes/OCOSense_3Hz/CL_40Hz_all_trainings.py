##########################################################
# Import libraries
##########################################################
print('\nImporting libraries...')
import os
import gc
import json
from STResNet import Classifier
import numpy as np
import pickle
import pandas as pd
import utils
import warnings
import time
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import config_CL_40Hz_all as cfg




##########################################################
# Set Warnings - GPU - Threads
##########################################################
warnings.filterwarnings('ignore')
#The levels are: 0: Display all logs (default behavior).1: Display all logs except debug logs.2: Display only warning and error logs.3: Display only error logs.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['DEEPREG_LOG_LEVEL'] = '5'

# setting parallelism
# these lines are restricting parallelism for both OpenMP and TensorFlow. They ensure that both libraries use only one thread,
# effectively running operations serially. This can be useful for debugging or for environments where multi-threading might cause issues.
# However, it can also reduce performance, especially on multi-core machines, as it doesn't take advantage of the available parallelism.
os.environ["OMP_NUM_THREADS"] = str(cfg.NUM_THREADS)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(cfg.NUM_THREADS)
os.environ["TF_NUM_INTEROP_THREADS"] = str(cfg.NUM_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(cfg.NUM_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(cfg.NUM_THREADS)
tf.config.set_soft_device_placement(True)

# GPU Memory Management
# TensorFlow, by default, allocates all available GPU memory when it starts. This can be problematic if you're running multiple models
# or tasks on the same GPU. config.gpu_options.allow_growth = True ensures that TensorFlow starts by allocating only a small amount of 
# GPU memory and then increases it as needed. This prevents TensorFlow from hogging all the GPU memory. The tf.compat.v1.Session and 
# tf.compat.v1.keras.backend.set_session lines are setting this configuration for the TensorFlow session. 
utils.set_session(cfg.USE_GPU, cfg.GPU_DEVICE)





##########################################################
# DATA LOADING
##########################################################
# LOAD DATA
print('\nLoading data...')
# USI-HEAR
windows2 = np.load("../../datasets/OCOSense_windows_5.npy")
clients2 = np.load("../../datasets/OCOSense_clients_5.npy")
labels2 = np.load("../../datasets/OCOSense_labels_5.npy")

# Print data shape
print('OCOSense shape: {} {} {} {}'.format(windows2.shape, clients2.shape, labels2.shape, np.unique(labels2)))

# metrics
METRICS = {
    "Loss_40": [],
    "Accuracy_40": [],
    "F1_40": [],
}




##########################################################
# TRAINING AND TESTING repeated N times
##########################################################
for i in range(cfg.N_REPETITIONS):
    print(f'\n\n\n\033[1mREPETITION {i+1}/{cfg.N_REPETITIONS}\033[0m\n\n\n')
    gc.collect()
    time_start = time.time()



    ##########################################################
    # SPLIT DATA
    ##########################################################
    # split the windows by client, create a list
    windows_train2, labels_train2, frequency_train2, windows_test2, labels_test2 = utils.split_clients_by_frequency_40_all(windows2, labels2, clients2, client_split=16, seed=cfg.SEEDS[i])

    # split train in train and val - take the first 6 clients for validation from each sensor combination
    windows_val2, labels_val2, frequency_val2 = utils.extract_validation_set(windows_train2, labels_train2, frequency_train2, 6)

    # keep only clients with 'acc' in sensors
    frequency, signals_to_keep = '40', 22
    windows_train2 = [windows_train2[i] for i in range(len(windows_train2)) if frequency in frequency_train2[i]]
    labels_train2 = [labels_train2[i] for i in range(len(labels_train2)) if frequency in frequency_train2[i]]
    windows_val2 = [windows_val2[i] for i in range(len(windows_val2)) if frequency in frequency_val2[i]]
    labels_val2 = [labels_val2[i] for i in range(len(labels_val2)) if frequency in frequency_val2[i]]

    # remove other signals from windows - use only acc
    windows_train2 = [utils.add_new_context_vect(windows_train2[i][:signals_to_keep]) for i in range(len(windows_train2))]
    windows_val2 = [utils.add_new_context_vect(windows_val2[i][:signals_to_keep]) for i in range(len(windows_val2))]
    windows_test2[frequency] = [utils.add_new_context_vect(windows_test2[frequency][i][:signals_to_keep]) for i in range(len(windows_test2[frequency]))]
    print("  Dimensions after removing other signals: ", windows_train2[0][0].shape)

    # Split train and test - Initialize KFold
    NUM_CLASSES = len(np.unique(np.concatenate(labels_train2)))
    NUM_CHANNELS = len(windows_train2[0])
    print("  n. TEST OCOSense:", len(windows_test2["40"]))
    print("  n. VAL OCOSense:", len(windows_val2))
    print("  n. TRAIN OCOSense:", len(windows_train2))

    # Merge windows and labels
    windows_test = {}
    labels_test = {}
    windows_train2 = utils.merge_user_window(windows_train2, randomize=True, seed=cfg.SEEDS[i])
    windows_val2 = utils.merge_user_window(windows_val2, randomize=True, seed=cfg.SEEDS[i])
    windows_test['40'] = utils.merge_user_window(windows_test2['40'], randomize=True, seed=cfg.SEEDS[i])
    labels_train2 = utils.merge_label(labels_train2, randomize=True, seed=cfg.SEEDS[i])
    labels_val2 = utils.merge_label(labels_val2, randomize=True, seed=cfg.SEEDS[i])
    labels_test['40'] = utils.merge_label(labels_test2['40'], randomize=True, seed=cfg.SEEDS[i])

    # remove 5 from label of dataset 2 to bring in the rage 0 5
    labels_train2 = labels_train2 - 5
    labels_val2 = labels_val2 - 5
    labels_test['40'] = labels_test['40'] - 5

    # Print some random context vector
    print('  Random context vector:')
    print('  ', windows_train2[-1][0:5], '...')


    ##########################################################
    # TRAINING
    ##########################################################
    SIGNAL_SIZE = windows_train2[0][0].shape[1]
    # Parameters
    print('  OCOSense TRAINING (40Hz_all)')
    print("  Number of channels: ", NUM_CHANNELS)
    print("  Number of classes: ", NUM_CLASSES)
    print(f'  Signal Size: {SIGNAL_SIZE}')
    print("  Learning rate: ", cfg.LEARNING_RATE)
    print("  Loss: ", cfg.LOSS)
    print("  Batch size: ", cfg.BATCH_SIZE)
    print("  Number of epochs: ", cfg.NUM_EPOCHS)
    print('\n\n')

    #set model configuration
    window_length = 10 #seconds
    signal_sizes = [40*window_length]*22 #in this case the signal size is the same for all channels, but the model aloows different sizes
    sampling_rates = [40]*22  #in this case the sampling rate is the same for all channels, but the model aloows different sizes
    fft_win_len_factor = 6 #len of fft windows in seconds
    fft_win_length = []
    hop_length = []
    
    for k in range(len(signal_sizes)):
            fft_win_length.append(sampling_rates[k]*fft_win_len_factor)
            hop_length.append(int(fft_win_length[k]//4))
    
    model_config = []
    with open("./STResNet_config.json") as json_file: 
            model_config = json.load(json_file)
            model_config['signal_size'] = signal_sizes
            model_config['num_channels'] = len(signal_sizes)
            model_config['sampling_rate'] = sampling_rates
            model_config['fft_win_length'] = fft_win_length
            model_config['hop_length'] = hop_length
            model_config['max_filters'] = 32
            model_config['num_filters'] = 16
            model_config['num_res_blocks'] = 4
            model_config['pool_size'] = 2

    # create the model
    STResNet_model = Classifier(model_config, NUM_CLASSES, verbose=True).model #create the model

    # Load and compile model for server-side parameter evaluation
    STResNet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
        loss=cfg.LOSS,
        metrics=['accuracy', utils.f1_m])

    # define learning rate scheduler 
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(cfg.scheduler)

    # Define the EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=cfg.PATIENCE, 
        verbose=1)

    # define checkpoint 
    model_save_path = cfg.MODELS_FOLDER + f"40Hz_all_model_CL_R{i}.h5"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_save_path, 
        monitor='val_accuracy', 
        save_best_only=True, 
        verbose=cfg.CALLBACK_VERBOSE, 
        mode='max',
        save_weights_only=True
    )

    history = STResNet_model.fit(
        windows_train2, labels_train2, 
        validation_data=(windows_val2, labels_val2), 
        epochs=cfg.NUM_EPOCHS, 
        batch_size=cfg.BATCH_SIZE, 
        verbose=cfg.TRAINING_VERBOSE,
        callbacks=[reduce_lr, early_stopping, model_checkpoint]
    )

    train_loss = history.history['loss']
    train_accuracy = history.history['accuracy']
    train_f1 = history.history['f1_m']
    val_loss = history.history['val_loss']
    val_accuracy = history.history['val_accuracy']
    val_f1 = history.history['val_f1_m']

    # Save the training history as a CSV file
    history_df = pd.DataFrame({
        'Epoch': list(range(len(train_loss))),
        'Loss': train_loss,
        'Accuracy': train_accuracy,
        'F1': train_f1,
        'Loss val': val_loss,
        'Accuracy val': val_accuracy,
        'F1 val': val_f1,   
    })
    history_df.to_csv(cfg.LOGDIR + f"training_history_OCO_40Hz_all_CL_R{i}.csv", index=False)

    # plot 
    utils.plot_trends(train_loss, train_accuracy, train_f1, title=f'Training_OCO_40Hz_all_CL_R{i}', fold=cfg.IMAGES_FOLDER)
    utils.plot_trends(val_loss, val_accuracy, val_f1, title=f'Validation_OCO_40Hz_all_CL_R{i}', fold=cfg.IMAGES_FOLDER)




    ##########################################################
    # TESTING
    ##########################################################
    # Clear garbage collector 
    gc.collect()
    print('\n  OCOSense TESTING (40Hz_all)')

    # Load the best model
    STResNet_model.load_weights(cfg.MODELS_FOLDER + f"40Hz_all_model_CL_R{i}.h5")

    # Evaluate the model on the test set
    test_loss40, test_accuracy40, test_f140 = cfg.test_on_one_dataset_mean(STResNet_model, windows_test['40'], labels_test['40'])

    # Print the results
    print(f"Test Loss (40Hz_all): {test_loss40:.4f}")
    print(f"Test Accuracy (40Hz_all): {test_accuracy40:.4f}")
    print(f"Test F1 (40Hz_all): {test_f140:.4f}")


    # save results
    METRICS["Loss_40"].append(test_loss40)
    METRICS["Accuracy_40"].append(test_accuracy40)
    METRICS["F1_40"].append(test_f140)

    with open(cfg.LOGDIR + f"metrics_OCO_40Hz_all_CL_R{i}.pkl", 'wb') as f:
        pickle.dump(METRICS, f)




    ##########################################################
    # UPDATE TIME
    ##########################################################
    time_last_rep = time.time() - time_start
    estimated_finish_time = time_last_rep * (cfg.N_REPETITIONS - (i + 1))
    days = estimated_finish_time // 86400
    remaining_sec_after_days = estimated_finish_time % 86400
    print('\n\n\n')
    print(f"\033[91mDuration of the last repetition: {time.strftime('%H:%M:%S', time.gmtime(time_last_rep))}\033[0m")
    print(f"\033[91mEstimated finish time: {days} days, {time.strftime('%H:%M:%S', time.gmtime(remaining_sec_after_days))}\033[0m")




##########################################################
# SAVE RESULTS
##########################################################
with open(cfg.LOGDIR + f"metrics_OCO_40Hz_all_CL_REP.pkl", 'wb') as f:
    pickle.dump(METRICS, f)




##########################################################
# PRINT RESULTS
##########################################################
print("\n\n\n\033[91mMETRICS\033[0m")
print("OCOSense")
print("Accuracy (40Hz_all): {:.2f} ± {:.2f}".format(np.mean(METRICS["Accuracy_40"])*100, np.std(METRICS["Accuracy_40"])*100))
print("F1 (40Hz_all): {:.2f} ± {:.2f}".format(np.mean(METRICS["F1_40"])*100, np.std(METRICS["F1_40"])*100))
print("Loss (40Hz_all): {:.3f} ± {:.3f}".format(np.mean(METRICS["Loss_40"]), np.std(METRICS["Loss_40"])))

