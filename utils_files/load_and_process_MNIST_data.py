"""
Script to hold functions related to training data and model building for creating a MNIST classifier.
"""


import sys
from pathlib import Path
import time
import keras
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

# add the parent directory of this file to the sys.path so that utils can be imported if the script is run from utils directory
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils_files.model_training_utils import Training_Parameters_and_Info


def load_and_process_MNIST_data(train_params_and_info: Training_Parameters_and_Info):
    """
    load the MNIST data, scale it, and split it up into train, validation, and test sets
    
    Parameters:
        train_params_and_info: Training_Parameters_and_Info object containing the parameters for the training run.  Used in this function to specify...
            kfold_index: int, index of the fold to use for training and validation data
            debug_run: bool, if True, use a minimal subset of the data for quick testing code
            small_train_set: bool, if True, use a small subset of the training data to see how the model trains with limited data
    Returns:
        x_train: numpy array of shape (num_train_samples, 28, 28, 1)
        x_val: numpy array of shape (num_val_samples, 28, 28, 1)
        x_test: numpy array of shape (num_test_samples, 28, 28, 1)
        y_train: numpy array of shape (num_train_samples,)
        y_val: numpy array of shape (num_val_samples,)
        y_test: numpy array of shape (num_test_samples,)
    """
    print(f"Loading the MNIST data...")
    
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # combine the train and test sets so they can be split up later
    all_data = np.concatenate((x_train, x_test), axis=0)
    all_labels = np.concatenate((y_train, y_test), axis=0)
    
    # scale the images to the [0, 1] range
    all_data = all_data.astype("float32") / 255
    # expand the shape of the samples by adding a channel dimension to the end changing each sample from (28, 28) to (28, 28, 1).  This doesn't
    # change the data, but puts each pixel value in a 1x1 array, as having a channel dimension is a requirement for Conv2D layers in Keras.
    all_data = np.expand_dims(all_data, -1)
    
    # First split: 80% remaining (training + validation) and 20% test
    x_remaining, x_test, y_remaining, y_test = train_test_split(
        all_data, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )
    
    # Create stratified KFold cross-validator with 4 splits and select the fold to use
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    kfold_index = train_params_and_info.kfold_index
    # Split the remaining data into training and validation sets using the specified fold
    for i, (train_index, val_index) in enumerate(skf.split(x_remaining, y_remaining)):
        if i == kfold_index:
            x_train, x_val = x_remaining[train_index], x_remaining[val_index]
            y_train, y_val = y_remaining[train_index], y_remaining[val_index]
            break
    
    num_of_classes = len(np.unique(y_train))
    input_shape = x_train.shape[1:]  # input shape for MNIST is (28, 28, 1)
    
    if train_params_and_info.debug_run and train_params_and_info.small_train_set:
        raise ValueError("Cannot use both debug_run and small_train_set together.  Please select only one.")
    
    # if debug_run selected, use a minimum subset of the data for very fast training and testing so the code can be tested quickly
    if train_params_and_info.debug_run:
        for i in range(10):
            print("|DEBUG RUN|------------- running with a very limited number of training and testing samples to quickly test code -------------|DEBUG RUN|")
            time.sleep(0.05 * i)
        print()
        
        sample_count = 10
        x_train = x_train[:sample_count]
        y_train = y_train[:sample_count]
        x_val = x_val[:sample_count]
        y_val = y_val[:sample_count]
        x_test = x_test[:sample_count]
        y_test = y_test[:sample_count]
        
        # use all_data to determine the number of classes as this subset may not have all classes
        num_of_classes = len(np.unique(all_labels))
    
    # if small_train_set selected, use a small subset of the data to check how training is working with a limited dataset
    elif train_params_and_info.small_train_set:
        for i in range(10):
            print("|SMALL TRAIN SET|------------- running with a small number of training samples to quickly test code -------------|SMALL TRAIN SET|")
            time.sleep(0.05 * i)
        print()
        
        sample_count = 256
        x_train = x_train[:sample_count]
        y_train = y_train[:sample_count]
        x_val = x_val[:sample_count]
        y_val = y_val[:sample_count]
    
    # determine the number of classes and input shape from the data
    print(f"Number of classes: {num_of_classes}")
    print(f"Input sample shape: {input_shape}")
    print()
    
    # Verify the shapes of the datasets and distributions of labels
    print(f"Training data shape:   {x_train.shape}")
    print(f"Validation data shape: {x_val.shape}")
    print(f"Test data shape:       {x_test.shape}")
    print()
    
    print(f"Training labels shape:   {y_train.shape}")
    print(f"Validation labels shape: {y_val.shape}")
    print(f"Test labels shape:       {y_test.shape}")
    print()
    
    print(f"All data labels distribution:   {np.bincount(all_labels)}")
    print(f"Training labels distribution:   {np.bincount(y_train)}")
    print(f"Validation labels distribution: {np.bincount(y_val)}")
    print(f"Test labels distribution:       {np.bincount(y_test)}")
    print()
    return x_train, x_val, x_test, y_train, y_val, y_test, num_of_classes, input_shape
