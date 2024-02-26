# Import libraries
import os
import numpy as np
from tensorflow.keras import backend as K
import json
from tensorflow.autograph import experimental
from scipy.signal import resample
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# Metrics
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

@experimental.do_not_convert
def f1_m(y_true, y_pred):
    # Used only during training, then sklearn f1_score is used
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Plotting metrics over epochs
def plot_trends(train_loss, train_accuracy, train_f1, title, fold):
    # Convert the data into pandas DataFrames
    df = pd.DataFrame({
        'Epoch': list(range(len(train_loss))),
        'Loss': train_loss,
        'Accuracy': train_accuracy,
        'F1': train_f1
    })

    # Find the epoch with the maximum accuracy
    max_accuracy_epoch = df['Accuracy'].idxmax()
    max_accuracy_value = df['Accuracy'].max()

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot accuracy and F1 in the first subplot
    sns.lineplot(data=df, x='Epoch', y='Accuracy', ax=axes[0], color='red', label='Accuracy')
    # sns.lineplot(data=df, x='Epoch', y='F1', ax=axes[0], color='green', label='F1')
    axes[0].scatter(max_accuracy_epoch, max_accuracy_value, color='blue', s=100, marker='*', label='Max Accuracy')
    axes[0].set_title(title + ' - Accuracy and F1 over Epochs')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    
    # Plot loss in the second subplot
    sns.lineplot(data=df, x='Epoch', y='Loss', ax=axes[1], color='blue')
    axes[1].set_title(title + ' - Loss over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    plt.tight_layout()

    # Save the figure
    plt.savefig(fold + title + '.png')  # You can change the file name and format as needed
    # Optionally, you can close the figure after saving to free up memory
    plt.close(fig)

# Extract validation set clients from training set
def extract_validation_set(windows_train, labels_train, sensors_train, N_clients_per_frequency=2):
    windows_val = []
    labels_val = []
    sensors_val = []
    
    sensor_counts = {'40': 0, '3': 0, }
    i = 0
    while i < len(sensors_train):
        sensor = sensors_train[i]
        if sensor_counts[sensor] < N_clients_per_frequency:
            # Pop and append to validation sets
            windows_val.append(windows_train.pop(i))
            labels_val.append(labels_train.pop(i))
            sensors_val.append(sensors_train.pop(i))

            sensor_counts[sensor] += 1
        else:
            i += 1

    return windows_val, labels_val, sensors_val

# Introduction of the context vector
def add_new_context_vect(window):
    new_context = [1]*len(window)
    new_context = np.tile(new_context, (window[0].shape[0], 1))
    window.append(new_context)
    return window

# Split channels by frequency
def split_channels_by_frequency(features, frequency=None):
    assert frequency!=None, "Please specify the sensors to use"
    n_channels = features.shape[2]
    if frequency == "40":
        # place 1 where A is present in features_OCO, else 0
        context = [1]*n_channels + [0]*n_channels
        context = np.tile(context, (features.shape[0], 1))
        missing_data = np.zeros((features.shape[0], int(features.shape[1]/13), n_channels))
        missing_data = np.concatenate([missing_data],axis=2)
        missing_data = np.split(missing_data, missing_data.shape[2], axis=2)
        features = np.concatenate([features],axis=2)
        features = np.split(features, features.shape[2], axis=2)
        features = features + missing_data
    elif frequency == "3":
        # place 1 where A or G is present in features_OCO, else 0
        context =  [0]*n_channels + [1]*n_channels
        context = np.tile(context, (features.shape[0], 1))
        missing_data = np.zeros((features.shape[0], features.shape[1], n_channels))
        missing_data = np.concatenate([missing_data],axis=2)
        missing_data = np.split(missing_data, missing_data.shape[2], axis=2)
        # resample of the signal to 3Hz
        features = resample(features, num=int(features.shape[1]/13), axis=1)
        features = np.concatenate([features],axis=2)
        features = np.split(features, features.shape[2], axis=2)
        features = missing_data + features
        
    features.append(context)
    return features

# Split clients by frequency
def split_clients_by_frequency(windows, labels, clients, seed=42):
    # split the windows by client, create a list
    windows_clients = []
    labels_clients = []
    frequency_clients = []
    windows_clients_test = {"40":[], "3":[]}
    labels_clients_test = {"40":[], "3":[]}
    # randomly shuffle clients
    random_clients = np.unique(clients)
    # fix the seed
    np.random.seed(seed)
    np.random.shuffle(random_clients)
    # add the seed tp the print with random shuffled clients
    print(f"  Random shuffled clients (seed: {seed}): {random_clients}")
    
    # divide in 4 groups of 5 clients
    f40, f3 = random_clients[:10], random_clients[10:20], 
    test_user = random_clients[20:]
    print("  Clients in the test set: ", test_user)

    for client in np.unique(clients):
        # if clients only have accelerometer
        if client in f40:
            # find the indices of the client
            client_idx = np.where(clients == client)[0]
            np.random.shuffle(client_idx)
            windows_client = windows[client_idx]
            windows_client = split_channels_by_frequency(windows_client, frequency="40")
            # append the windows to the list
            windows_clients.append(windows_client)
            labels_clients.append(labels[client_idx])
            frequency_clients.append("40")
        # if clients have accelerometer and gyroscope
        elif client in f3:
            # find the indices of the client
            client_idx = np.where(clients == client)[0]
            np.random.shuffle(client_idx)
            windows_client = windows[client_idx]
            windows_client = split_channels_by_frequency(windows_client, frequency="3")
            # append the windows to the list
            windows_clients.append(windows_client)
            labels_clients.append(labels[client_idx])
            frequency_clients.append("3")
        # if clients are in the test
        elif client in test_user:
            # find the indices of the client
            client_idx = np.where(clients == client)[0]
            np.random.shuffle(client_idx)
            windows_client = windows[client_idx]
            # accelerometer
            w = split_channels_by_frequency(windows_client, frequency="40")
            windows_clients_test["40"].append(w)
            labels_clients_test["40"].append(labels[client_idx])
            # accelerometer and gyroscope
            w = split_channels_by_frequency(windows_client, frequency="3")
            windows_clients_test["3"].append(w)
            labels_clients_test["3"].append(labels[client_idx])
    
    return windows_clients, labels_clients, frequency_clients, windows_clients_test, labels_clients_test

# Set session - GPU
def set_session(use_gpu=True, gpu_device=0): 
    if not use_gpu:
        # Hide GPU from TensorFlow
        print('Hiding GPU from TensorFlow')
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        # show GPU device
        print('Showing GPU to TensorFlow')
        #os.environ["CUDA_VISIBLE_DEVICES"] = config_CL.GPU_DEVICE
        physical_devices = tf.config.list_physical_devices('GPU')
        physical_devices = [physical_devices[gpu_device]]
        tf.config.set_visible_devices(physical_devices, 'GPU')
    print('Visible Physical devices:',physical_devices)
    for gpu in physical_devices:
           print('GPU used:', gpu)
           tf.config.experimental.set_memory_growth(gpu, True) #allocte memory as needed
    FLOAT_PRECISION = 'float32' #use 32 instead of 64 to save on memory
    policy = tf.keras.mixed_precision.Policy(FLOAT_PRECISION)
    tf.keras.mixed_precision.set_global_policy(policy)
    tf.keras.backend.set_floatx(FLOAT_PRECISION)
    cpu_device = tf.config.list_logical_devices('CPU')[0]
    gpu_devices = tf.config.list_logical_devices('GPU') 

def split_channels_by_frequency_down(features, frequency=None):
    assert frequency!=None, "Please specify the sensors to use"
    n_channels = features.shape[2]
    if frequency == "3":
        # place 1 where A or G is present in features_OCO, else 0
        context = [1]*n_channels
        context = np.tile(context, (features.shape[0], 1))
        # resample of the signal to 5Hz
        features = resample(features, num=int(features.shape[1]/13), axis=1)
        features = np.concatenate([features],axis=2)
        features = np.split(features, features.shape[2], axis=2)
        
    features.append(context)
    return features

def split_clients_by_frequency_down(windows, labels, clients, seed=42):
    # split the windows by client, create a list
    windows_clients = []
    labels_clients = []
    frequency_clients = []
    windows_clients_test = {"3":[]}
    labels_clients_test = { "3":[]}
    # randomly shuffle clients
    random_clients = np.unique(clients)
    # fix the seed
    np.random.seed(seed)
    np.random.shuffle(random_clients)
    # add the seed tp the print with random shuffled clients
    print(f"  Random shuffled clients (seed: {seed}): {random_clients}")
    
    # divide in 4 groups of 5 clients
    f3 = random_clients[:20] 
    test_user = random_clients[20:]
    print("  Clients in the test set: ", test_user)

    for client in np.unique(clients):
        # if clients only have accelerometer
        if client in f3:
            # find the indices of the client
            client_idx = np.where(clients == client)[0]
            np.random.seed(seed)
            np.random.shuffle(client_idx)
            windows_client = windows[client_idx]
            windows_client = split_channels_by_frequency_down(windows_client, frequency="3")
            # append the windows to the list
            windows_clients.append(windows_client)
            labels_clients.append(labels[client_idx])
            frequency_clients.append("3")
        # if clients are in the test
        elif client in test_user:
            # find the indices of the client
            client_idx = np.where(clients == client)[0]
            np.random.seed(seed)
            np.random.shuffle(client_idx)
            windows_client = windows[client_idx]
            # accelerometer and gyroscope
            w = split_channels_by_frequency_down(windows_client, frequency="3")
            windows_clients_test["3"].append(w)
            labels_clients_test["3"].append(labels[client_idx])
    
    return windows_clients, labels_clients, frequency_clients, windows_clients_test, labels_clients_test

def split_channels_by_frequency_40down(features, frequency=None):
    assert frequency!=None, "Please specify the sensors to use"
    n_channels = features.shape[2]
    if frequency == "40":
        # place 1 where A is present in features_OCO, else 0
        context = [1]*n_channels + [1]*n_channels
        context = np.tile(context, (features.shape[0], 1))
        features_down = resample(features, num=int(features.shape[1]/13), axis=1)
        features_down = np.concatenate([features_down],axis=2)
        features_down = np.split(features_down, features_down.shape[2], axis=2)
        features = np.concatenate([features],axis=2)
        features = np.split(features, features.shape[2], axis=2)
        features = features + features_down
    elif frequency == "3":
        # place 1 where A or G is present in features_OCO, else 0
        context =  [0]*n_channels + [1]*n_channels
        context = np.tile(context, (features.shape[0], 1))
        missing_data = np.zeros((features.shape[0], features.shape[1], n_channels))
        missing_data = np.concatenate([missing_data],axis=2)
        missing_data = np.split(missing_data, missing_data.shape[2], axis=2)
        # resample of the signal to 5Hz
        features = resample(features, num=int(features.shape[1]/13), axis=1)
        features = np.concatenate([features],axis=2)
        features = np.split(features, features.shape[2], axis=2)
        features = missing_data + features
        
    features.append(context)
    return features

def split_clients_by_frequency_40_down(windows, labels, clients, seed=42):
    # split the windows by client, create a list
    windows_clients = []
    labels_clients = []
    frequency_clients = []
    windows_clients_test = {"40":[], "3":[]}
    labels_clients_test = {"40":[], "3":[]}
    # randomly shuffle clients
    random_clients = np.unique(clients)
    # fix the seed
    np.random.seed(seed)
    np.random.shuffle(random_clients)
    # add the seed tp the print with random shuffled clients
    print(f"  Random shuffled clients (seed: {seed}): {random_clients}")
    
    # divide in 4 groups of 5 clients
    f40, f3 = random_clients[:10], random_clients[10:20], 
    test_user = random_clients[20:]
    print("  Clients in the test set: ", test_user)

    for client in np.unique(clients):
        # if clients only have accelerometer
        if client in f40:
            # find the indices of the client
            client_idx = np.where(clients == client)[0]
            np.random.shuffle(client_idx)
            windows_client = windows[client_idx]
            windows_client = split_channels_by_frequency_40down(windows_client, frequency="40")
            # append the windows to the list
            windows_clients.append(windows_client)
            labels_clients.append(labels[client_idx])
            frequency_clients.append("40")
        # if clients have accelerometer and gyroscope
        elif client in f3:
            # find the indices of the client
            client_idx = np.where(clients == client)[0]
            np.random.shuffle(client_idx)
            windows_client = windows[client_idx]
            windows_client = split_channels_by_frequency_40down(windows_client, frequency="3")
            # append the windows to the list
            windows_clients.append(windows_client)
            labels_clients.append(labels[client_idx])
            frequency_clients.append("3")
        # if clients are in the test
        elif client in test_user:
            # find the indices of the client
            client_idx = np.where(clients == client)[0]
            np.random.shuffle(client_idx)
            windows_client = windows[client_idx]
            # accelerometer
            w = split_channels_by_frequency_40down(windows_client, frequency="40")
            windows_clients_test["40"].append(w)
            labels_clients_test["40"].append(labels[client_idx])
            # accelerometer and gyroscope
            w = split_channels_by_frequency_40down(windows_client, frequency="3")
            windows_clients_test["3"].append(w)
            labels_clients_test["3"].append(labels[client_idx])
    
    return windows_clients, labels_clients, frequency_clients, windows_clients_test, labels_clients_test

# Merge windows from list of clients (data prepared for FL)
def merge_user_window(window, randomize=False, seed=42):
    final_matrices = []
    for i in range(len(window[0])):
        matrices_to_concatenate = [user[i] for user in window]
        concatenated_matrix = np.concatenate(matrices_to_concatenate, axis=0)
        if randomize:
            np.random.seed(seed)
            np.random.shuffle(concatenated_matrix)
        final_matrices.append(concatenated_matrix)

    return final_matrices

# Merge labels from list of clients (data prepared for FL)
def merge_label(label, randomize=False, seed=42):
    matrices_to_concatenate = [user for user in label]
    concatenated_matrix = np.concatenate(matrices_to_concatenate, axis=0)
    if randomize:
        np.random.seed(seed)
        np.random.shuffle(concatenated_matrix)
    return concatenated_matrix

def split_clients_by_frequency_40_all(windows, labels, clients, seed=42):
    # split the windows by client, create a list
    windows_clients = []
    labels_clients = []
    frequency_clients = []
    windows_clients_test = {"40":[]}
    labels_clients_test = {"40":[]}
    # randomly shuffle clients
    random_clients = np.unique(clients)
    # fix the seed
    np.random.seed(seed)
    np.random.shuffle(random_clients)
    # add the seed tp the print with random shuffled clients
    print(f"  Random shuffled clients (seed: {seed}): {random_clients}")
    
    # divide in 4 groups of 5 clients
    f40 = random_clients[:20] 
    test_user = random_clients[20:]
    print("  Clients in the test set: ", test_user)

    for client in np.unique(clients):
        # if clients only have accelerometer
        if client in f40:
            # find the indices of the client
            client_idx = np.where(clients == client)[0]
            np.random.shuffle(client_idx)
            windows_client = windows[client_idx]
            windows_client = split_channels_by_frequency(windows_client, frequency="40")
            # append the windows to the list
            windows_clients.append(windows_client)
            labels_clients.append(labels[client_idx])
            frequency_clients.append("40")
        # if clients have accelerometer and gyroscope
        # if clients are in the test
        elif client in test_user:
            # find the indices of the client
            client_idx = np.where(clients == client)[0]
            np.random.shuffle(client_idx)
            windows_client = windows[client_idx]
            # accelerometer
            w = split_channels_by_frequency(windows_client, frequency="40")
            windows_clients_test["40"].append(w)
            labels_clients_test["40"].append(labels[client_idx])
    
    return windows_clients, labels_clients, frequency_clients, windows_clients_test, labels_clients_test
