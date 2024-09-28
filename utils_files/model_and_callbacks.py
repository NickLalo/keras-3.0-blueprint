"""
Script to hold functions related to model building and callbacks.
"""


import os
import sys
from pathlib import Path
import time
import csv
import yaml
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import mlflow


# add the parent directory of this file to the sys.path so that utils can be imported if the script is run from utils directory
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils_files.model_training_utils import Training_Parameters_and_Info, plot_training_and_validation_loss, jax_gpu_check, \
    create_placeholder_training_and_validation_loss_plot
from utils_files.utils import convert_seconds_to_readable_time_str


# hardcoded paths and names
MLFLOW_CONFIG_FILE = "_configs/mlflow_configs.yaml"


def build_and_compile_model(input_shape, num_of_classes, train_params_and_info: Training_Parameters_and_Info):
    """
    build and compile a simple CNN model for MNIST classification
    
    Parameters:
        input_shape: tuple representing the shape of the input data (height, width, channels)
        num_of_classes: int representing the number of classes in the classification task
        train_params_and_info: Training_Parameters_and_Info object containing the parameters and info for the training run.
            learning_rate (float): the learning rate for the optimizer
            dropout_rate (float): the dropout rate for the dropout layer
    Returns:
        model: keras model object
    """
    # Reset the Keras session to clear previous model and layer names.  Without this step, the model summary will auto increment the layer names
    # for every subsequent model that is created in the grid search, which can be confusing.
    keras.backend.clear_session()
    
    # depending on which backend is being used, check to see if a GPU is available and set the mixed precision policy if it is
    keras_backend = keras.backend.backend()
    if keras_backend == "jax":
        gpu_available = jax_gpu_check()
    elif keras_backend == "tensorflow":
        gpu_available = tf.config.list_physical_devices("GPU")
    else:
        # PyTorch backend not supported, but that could possibly be an easy fix if needed
        raise ValueError(f"Unknown backend: {keras_backend}")
    
    # set the mixed precision policy if a GPU is being used.  This can greatly speed up training on a tensor core GPU, but may not speed up training 
    # on all GPUs.  However, setting this policy could cause the model to train slower on a non-tensor core GPU, so it is best to test both ways.
    if gpu_available:
        print("GPU detected, attempting to set mixed precision policy to 'mixed_float16' for faster training on tensor core GPUs")
        keras.mixed_precision.set_global_policy("mixed_float16")
    
    # define the model
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(train_params_and_info.dropout_rate),
            keras.layers.Dense(num_of_classes, activation="softmax"),
        ]
    )
    
    # compile the model with a loss function, optimizer, and metrics
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=train_params_and_info.learning_rate),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
        ],
    )
    
    # save the model summary to a txt file
    train_params_and_info.save_model_summary_to_file(model, num_of_classes)
    # save the model architecture (as a graph) to a png file
    train_params_and_info.save_model_architecture_to_file(model)
    
    return model


def get_model_checkpoint_callback(train_params_and_info: Training_Parameters_and_Info):
    """
    create a ModelCheckpoint callback to save a new copy of the model each time the validation loss improves
    
    Parameters:
        train_params_and_info: Training_Parameters_and_Info object containing the parameters and info for the training run
            model_and_metrics_dir: PurePath obj representing the directory to save the best model
    Returns:
        checkpoint_callback: keras.callbacks.ModelCheckpoint object
        train_params_and_info: Training_Parameters_and_Info object
    """
    # set and keep track of the path to the model checkpoint directory.
    train_params_and_info.checkpoint_dir = train_params_and_info.model_and_metrics_dir.joinpath("model_checkpoints")
    
    # create the model checkpoint callback
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        # use nested curly braces to specify placeholders for the epoch and val_loss that will be filled in during training
        filepath=f"{train_params_and_info.checkpoint_dir}/best_model-epoch_{{epoch:03d}}-val_loss_{{val_loss:.2f}}.keras",
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
    )
    return checkpoint_callback, train_params_and_info


class Clean_Up_Checkpoints_Callback(keras.callbacks.Callback):
    """
    Custom Keras callback that deletes all but the best (last saved) model checkpoint after training is completed.  This is an essential step
    for running a random grid search where many models are saved and space is limited.
    
    Attributes:
        checkpoint_dir (PurePath): The directory where the model checkpoints are saved.
    """
    def __init__(self, train_params_and_info: Training_Parameters_and_Info):
        """
        Initializes the Clean_Up_Checkpoints_Callback.
        
        Args:
            train_params_and_info: Training_Parameters_and_Info object containing the parameters and info for the training run
                checkpoint_dir: PurePath object representing the directory where the model checkpoints are saved
        """
        super(Clean_Up_Checkpoints_Callback, self).__init__()
        self.checkpoint_dir = train_params_and_info.checkpoint_dir
        # save a reference to the train_params_and_info object so that the on_train_end method can access it
        self.train_params_and_info = train_params_and_info
        return
    
    def on_train_end(self, logs=None):
        """
        Called at the end of training. Deletes all but the best (last saved) model checkpoint.
        Models are only saved when the validation loss improves, so by sorting the list, the best model is the last one saved.
        Args:
            logs (dict, optional): Currently, this argument is ignored.
        """
        # get a list of all the model saves in the checkpoint directory
        model_saves_list = os.listdir(self.checkpoint_dir)
        # sort the list of model saves so that the best model is the last one saved
        model_saves_list.sort()
        
        # remove all but the last model (the best model)
        for model_save in model_saves_list[:-1]:
            # remove all model checkpoints that were saved on the way to the best model
            model_to_delete = self.checkpoint_dir.joinpath(model_save)
            os.remove(model_to_delete)
        
        # save the last model name to the train_params_and_info object because it is the best model
        best_model_path = self.checkpoint_dir.joinpath(model_saves_list[-1])
        # NOTE: if the Clean_Up_Checkpoints_Callback is not being used (like when training a GAN), this calculation of best model path needs to be
        # moved elsewhere, otherwise later code that tries to access train_params_and_info.best_model_path will see the value as None
        self.train_params_and_info.best_model_path = best_model_path
        return


def get_early_stopping_callback(train_params_and_info: Training_Parameters_and_Info):
    """
    create an EarlyStopping callback to stop training if the validation loss does not improve
    
    Parameters:
        train_params_and_info: Training_Parameters_and_Info object containing the parameters and info for the training run
            early_stopping_patience: int representing the number of epochs to wait for improvement before stopping training
    Returns:
        early_stopping_callback: keras.callbacks.EarlyStopping object
    """
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        verbose=1,
        patience=train_params_and_info.early_stopping_patience,
    )
    return early_stopping_callback


def get_reduce_lr_on_plateau_callback():
    """
    create a ReduceLROnPlateau callback to reduce the learning rate if the validation loss does not improve
    Returns:
        reduce_lr_on_plateau_callback: keras.callbacks.ReduceLROnPlateau object
    """
    reduce_lr_on_plateau_callback = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,  # new learning rate = old learning rate * factor
        patience=5,  # this should be a lower value than the early stopping patience so that the learning rate can be adjusted before stopping
        verbose=1,
    )
    return reduce_lr_on_plateau_callback


class Custom_MLFlow_Logger(keras.callbacks.Callback):
    """
    A custom Keras callback to log training and validation metrics to MLFlow.
    """
    def __init__(self, train_params_and_info: Training_Parameters_and_Info):
        """
        Initializes the Custom_MLFlow_Logger callback.
        
        Args:
            train_params_and_info: Training_Parameters_and_Info object containing the parameters and info for the training run
                run_name: str representing the name of the run
        """
        # raise an error if the mlflow config file does not exist
        if not os.path.exists(MLFLOW_CONFIG_FILE):
            raise FileNotFoundError(f"The MLFlow config file {MLFLOW_CONFIG_FILE} does not exist, try running the script from the root directory.")
        
        # load the mlflow yaml file into a dictionary
        with open(MLFLOW_CONFIG_FILE, "r") as file:
            mlflow_configs = yaml.load(file, Loader=yaml.FullLoader)
        
        # set the mlflow tracking uri to the local storage path, this is where the mlflow logs will be saved
        mlflow_storage_path = mlflow_configs["mlflow_storage_path"]
        mlflow.set_tracking_uri(f"{mlflow_storage_path}")
        
        # set the experiment name
        experiment_name = mlflow_configs["experiment_name"]
        mlflow.set_experiment(experiment_name)
        
        # start the mlflow run with a specific run name that matches the subdirectory in the model_and_metrics_saves directory
        run_name = train_params_and_info.run_name
        mlflow.start_run(run_name=run_name, log_system_metrics=True)
        
        # log the training parameters to mlflow
        training_params_dict = train_params_and_info.get_training_parameters_dict()
        for key, value in training_params_dict.items():
            mlflow.log_param(key, value)
        
        # set a flag in the train_params_and_info object to indicate that MLFlow is being used to log metrics for this run
        train_params_and_info.using_mlflow = True
        return
    
    def on_epoch_end(self, epoch, logs=None):
        """
        logs all of the metrics being tracked in the training loop to MLFlow at the end of each epoch
        """
        for key, value in logs.items():
            mlflow.log_metric(key, value, step=epoch)
        return


class Time_History_Callback(keras.callbacks.Callback):
    """
    A custom Keras callback to estimate the time taken for each epoch during training including the overhead from callbacks. Can also 
    inform on the train time only (excluding the overhead from callbacks).
    
    Attributes:
        times (list): A list to store the time taken for each epoch.
        training_start_time (float): The time when training started.
        epoch_time_start (float): The time when an epoch started.
        total_training_time (float): The total time taken for training including the overhead from callbacks.
    """
    def __init__(self, train_params_and_info: Training_Parameters_and_Info):
        """
        Initializes the Time_History_Callback.
        
        Args:
            train_params_and_info: Training_Parameters_and_Info object containing the parameters and info for the training run
        """
        super(Time_History_Callback, self).__init__()
        self.epoch_train_times = []
        self.training_start_time = None
        self.epoch_time_start = None
        self.total_training_time = None
        self.epochs_trained = None
        
        # create a reference to the train_params_and_info object so that the on_train_end method can save time info about training to it
        self.train_params_and_info = train_params_and_info
        return
    
    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training. Initializes the list to store epoch times.
        
        Args:
            logs (dict, optional): Currently, this argument is ignored.
        """
        self.epoch_train_times = []
        self.training_start_time = time.time()
        return
    
    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the beginning of each epoch. Records the start time of the epoch.
        
        Args:
            epoch (int): The current epoch number.
            logs (dict, optional): Currently, this argument is ignored.
        """
        self.epoch_time_start = time.time()
        return
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch to records the end time of the epoch.
        
        Args:
            epoch (int): The current epoch number.
            logs (dict, optional): Currently, this argument is ignored.
        """
        # record the time taken for the epoch not including the overhead from callbacks
        epoch_time = time.time() - self.epoch_time_start
        self.epoch_train_times.append(epoch_time)
        return
    
    def on_train_end(self, logs=None):
        """
        Called at the end of training to calculate the total training time.
        
        Args:
            logs (dict, optional): Currently, this argument is ignored.
        """
        self.total_training_time = time.time() - self.training_start_time
        self.epochs_trained = len(self.epoch_train_times)
        
        # save the total train time as a readable string to the train_params_and_info object
        total_train_time_string = convert_seconds_to_readable_time_str(self.total_training_time)
        self.train_params_and_info.total_train_time_string = total_train_time_string
        self.train_params_and_info.total_train_time_seconds = self.total_training_time
        print(f"training completed in {total_train_time_string}\n")
        
        # save the average time per epoch as a readable string to the train_params_and_info object.  This includes the overhead from callbacks.
        # NOTE: if interested in the time per epoch without the overhead from callbacks, this can be done by taking the average of epoch_train_times
        time_per_epoch = self.total_training_time / self.epochs_trained
        time_per_epoch_string = convert_seconds_to_readable_time_str(time_per_epoch)
        self.train_params_and_info.time_per_epoch_string = time_per_epoch_string
        self.train_params_and_info.time_per_epoch_seconds = time_per_epoch
        return


class Custom_CSV_Logger(keras.callbacks.Callback):
    """
    A custom Keras callback to log training and validation metrics to a CSV file.  It also creates a training and validation loss plot that is updated
    after each epoch.  Monitoring this updating image can give an idea of how the training is progressing.
    
    Attributes:
        csv_filename (PurePath): The path to the CSV file where logs will be saved.
        sep (str): The separator used in the CSV file.
        append (bool): Whether to append to the existing CSV file or overwrite it.
        file (file object): The file object for the CSV file.
        writer (csv.writer): The CSV writer object.
        keys (list): The list of metric keys to log.
        append_header (bool): Whether to append the header to the CSV file.
    """
    def __init__(self, train_params_and_info: Training_Parameters_and_Info, separator=',', append=False):
        """
        Initializes the Custom_CSV_Logger callback.W
        
        Args:
            train_params_and_info (Training_Parameters_and_Info): Training_Parameters_and_Info object containing the parameters and info for the
                training run.
            parent_dir (PurePath): The directory where the CSV file will be saved.
            separator (str, optional): The separator used in the CSV file. Defaults to ','.
            append (bool, optional): Whether to append to the existing CSV file or overwrite it. Defaults to False.
        """
        super(Custom_CSV_Logger, self).__init__()
        self.csv_filename = train_params_and_info.model_and_metrics_dir.joinpath('training_logs.csv')
        train_params_and_info.csv_filename = self.csv_filename
        self.sep = separator
        self.append = append
        self.file = None
        self.writer = None
        self.keys = None
        self.append_header = True
        # Use the Agg backend for Matplotlib so that plots are saved to file instead of displayed
        plt.switch_backend('Agg')
        create_placeholder_training_and_validation_loss_plot(self.csv_filename)
        return
    
    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training. Prepares the CSV file for logging by opening it and writing the header if necessary.
        
        Args:
            logs (dict, optional): Currently, this argument is ignored.
        """
        if self.append and os.path.exists(self.csv_filename):
            with open(self.csv_filename, 'r') as f:
                self.append_header = not bool(len(f.readline()))
        
        mode = 'a' if self.append else 'w'
        self.file = open(self.csv_filename, mode, newline='')
        self.writer = csv.writer(self.file, delimiter=self.sep)
        return
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch. Logs the metrics to the CSV file and updates the training/validation loss plot.
        
        Args:
            epoch (int): The current epoch number.
            logs (dict, optional): Dictionary of metrics from the current epoch.
        """
        logs = logs or {}
        
        if not self.keys:
            self.keys = sorted(logs.keys())
            if self.append_header:
                self.writer.writerow(['epoch'] + self.keys)
        
        # 1 added to epoch to account for 0-based indexing.  These values represent the metrics at the end of an epoch.
        row = [epoch+1] + [logs[k] for k in self.keys]
        self.writer.writerow(row)
        self.file.flush()
        
        # create an intermediate plot of the training and validation loss after each epoch
        plot_training_and_validation_loss(self.csv_filename, best_epoch=None)
        return
    
    def on_train_end(self, logs=None):
        """
        Called at the end of training. Closes the CSV file.
        
        Args:
            logs (dict, optional): Currently, this argument is ignored.
        """
        self.file.close()
        # NOTE: the final function call of plot_training_and_validation_loss happens in in the train_params_and_info.print_and_save_train_test_results
        #   method because it relies on the best_epoch value that is extracted from the saved model name.
        return


def get_callbacks_for_training(train_params_and_info: Training_Parameters_and_Info):
    """
    load callbacks for training the model
    
    Parameters:
        train_params_and_info: Training_Parameters_and_Info object containing the parameters and info for the training run
    Returns:
        callbacks_list: list of keras.callbacks objects
        checkpoint_dir: PurePath object representing the directory where the model checkpoints are saved
        time_history_callback: Time_History_Callback object.  Used after training to get the total training time and average time per epoch
    """
    # get the callbacks that will add functionality before, during, and after training
    checkpoint_callback, train_params_and_info = get_model_checkpoint_callback(train_params_and_info)
    clean_up_checkpoints_callback = Clean_Up_Checkpoints_Callback(train_params_and_info)
    early_stopping_callback = get_early_stopping_callback(train_params_and_info)
    mlflow_callback = Custom_MLFlow_Logger(train_params_and_info)
    time_history_callback = Time_History_Callback(train_params_and_info)
    csv_logger_callback = Custom_CSV_Logger(train_params_and_info)
    # only use the ReduceLROnPlateau callback if the flag is set to True
    if train_params_and_info.reduce_lr_on_plateau:
        reduce_lr_on_plateau_callback = get_reduce_lr_on_plateau_callback()
    
    # package up the callback into a list
    callbacks_list = [
        checkpoint_callback, 
        early_stopping_callback, 
        mlflow_callback, 
        time_history_callback, 
        csv_logger_callback, 
        clean_up_checkpoints_callback
    ]
    
    if train_params_and_info.reduce_lr_on_plateau:
        callbacks_list.append(reduce_lr_on_plateau_callback)
    
    return callbacks_list, train_params_and_info