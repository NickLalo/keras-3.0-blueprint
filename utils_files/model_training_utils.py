"""
Script to hold helper functions used in training, evaluating, and inferencing with models.
"""


from __future__ import annotations  # needed for type hinting while avoiding circular imports as mentioned below
import os
import random
import time 
import sys
import json
import subprocess
import platform
from typing import Optional
from pathlib import Path, PurePath
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, AnnotationBbox, VPacker
import keras
import numpy as np
import tensorflow as tf
import jax
import mlflow

# add the parent directory of this file to the sys.path so that utils can be imported if the script is run from utils directory
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils_files.utils import get_time_stamp, Terminal_Logger


# hardcoded paths and names
MODEL_AND_METRICS_DIRECTORY_NAME = "model_and_metrics_saves"


class Training_Parameters_and_Info():
    """
    custom class to hold the parameters used for a training a keras model and information about the training run.  This class is designed to act as
    an easily trackable "global-like" object to give clarity to the logical flow of code with easily trackable operations.  It abstracts away as many
    details as possible to make the code more readable and maintainable.
    """
    def __init__(self, parameters_dict):
        # ------------------------------------------------------ Training Parameter Extraction -------------------------------------------------------
        # keep track of the names of hyperparameters
        self.hyperparameter_names = list(parameters_dict.keys())
        
        # parameters related to the dataset
        self.kfold_index = int(parameters_dict["kfold_index"])
        self.debug_run = bool(parameters_dict["debug_run"])
        self.small_train_set = bool(parameters_dict["small_train_set"])
        
        # parameters related to the model and optimizer
        self.dropout_rate = float(parameters_dict["dropout_rate"])
        self.learning_rate = float(parameters_dict["learning_rate"])
        
        # parameters related to callbacks
        self.early_stopping_patience = int(parameters_dict["early_stopping_patience"])
        self.reduce_lr_on_plateau = bool(parameters_dict["reduce_lr_on_plateau"])
        
        # parameters related to the training loop
        self.batch_size = int(parameters_dict["batch_size"])
        self.epoch_limit = int(parameters_dict["epoch_limit"])
        
        # if debug_run is True, then set the epoch limit to 1 as we only want to test the code for errors
        if self.debug_run:
            self.epoch_limit = 1
            print(f"Debug run is set to True, changing the epoch limit to 1\n")
        
        # ---------------------------------------------------------- Setup for Info Storage ----------------------------------------------------------
        # define attributes that will hold information about the run.  These attributes will be populated later on in the script, but for now will be
        # initialized with the type defined and a NONE value.  This helps by giving type hinting and allowing for tracking of the attributes back to
        # this block of code where the initial set locations are specified in comments.
        
        # parameter set when running a random grid search in the random_grid_search.py script, otherwise defaults to False
        self.using_grid_search = False
        ############ parameters set in create_model_and_metrics_dir() method defined in this class below
        self.script_start_time: Optional[float] = None
        self.run_name: Optional[str] = None
        self.model_and_metrics_dir: Optional[PurePath] = None
        self.terminal_logger: Optional[Terminal_Logger] = None
        
        ############ parameters set in the get_callbacks_for_training() function defined in model_and_callbacks.py
        self.checkpoint_dir: Optional[PurePath] = None  # set get_model_checkpoint_callback() function
        self.best_model_path: Optional[PurePath] = None  # set in the Clean_Up_Checkpoints_callback on_train_end() method
        self.using_mlflow: Optional[bool] = None  # set to true in the Custom_MLFlow_logger __init__() method
        # readable formatted string for total model train time and time per epoch HH:MM:SS 00:00:00
        self.total_train_time_string: Optional[str] = None  # set in the Time_History_Callback on_train_end() method
        self.time_per_epoch_string: Optional[str] = None  # set in the Time_History_Callback on_train_end() method
        self.total_train_time_seconds: Optional[float] = None  # set in the Time_History_Callback on_train_end() method
        self.time_per_epoch_seconds: Optional[float] = None  # set in the Time_History_Callback on_train_end() method
        self.csv_filename: Optional[PurePath] = None  # set in the Custom_CSV_Logger __init__() method
        
        ############ parameters set in the print_and_save_train_test_results() method defined in this class below
        self.best_epoch: Optional[int] = None
        self.final_training_loss: Optional[float] = None
        self.final_validation_loss: Optional[float] = None
        ############ parameters set at the end of train_model function called in train_model.py script
        ############
        return
    
    def create_model_and_metrics_dir(self):
        """
        create a model and metrics directory for the current run.
        """
        self.script_start_time = time.time()
        # get a timestamp to use as a unique identifier for the run which will be used as a directory name for the model and metrics
        self.run_name = str(get_time_stamp())
        self.model_and_metrics_dir = PurePath(MODEL_AND_METRICS_DIRECTORY_NAME).joinpath(self.run_name)
        # ensure that the full path to the model and metrics directory exists
        os.makedirs(self.model_and_metrics_dir, exist_ok=True)
        print(f"model and metrics directory created at: {self.model_and_metrics_dir}")
        
        # create a log file to save the terminal output in the model and metrics directory
        terminal_output_log_filename = self.model_and_metrics_dir.joinpath("terminal_output_logs.txt")
        self.terminal_logger = Terminal_Logger(terminal_output_log_filename)
        
        print(f"setting random seeds for a reproducible run...")
        random.seed(112)
        np.random.seed(112)
        keras.utils.set_random_seed(112)
        tf.random.set_seed(112)
        jax.random.PRNGKey(112)
        
        # save the training parameters to a json file in the model and metrics directory
        self.save_params_as_json()
        return
    
    def get_training_parameters_dict(self):
        """
        Return the Training_Parameters object as a dictionary.
        
        Returns:
            training_params_dict (dict): dictionary of the Training_Parameters object
        """
        training_params_dict = {}
        for param_name in self.hyperparameter_names:
            training_params_dict[param_name] = getattr(self, param_name)
        
        return training_params_dict
    
    def save_params_as_json(self):
        """
        Save the Training_Parameters object to a JSON file.
        """
        training_parameters_save_path = self.model_and_metrics_dir.joinpath("training_parameters.json")
        
        # package up the Training_Parameters object as a dictionary
        training_params_dict = self.get_training_parameters_dict()
        
        with open(training_parameters_save_path, "w") as f:
            json.dump(training_params_dict, f, indent=4)
        print(f"training parameters saved to {training_parameters_save_path}")
        
        # also print the parameters to the terminal
        print(f"--------- Training Parameters ---------")
        for key, value in training_params_dict.items():
            print(f"{key}: {value}")
        print(f"---------------------------------------\n")
        return
    
    def save_model_summary_to_file(self, model: keras.Model, num_of_classes):
        """
        Save the model summary to a file
        
        Parameters:
            model (keras.Model): the model to save the summary of
            input_shape (tuple): the shape of the input data
            num_of_classes (int): the number of classes in the classification problem
            model_and_metrics_directory (PurePath): the directory to save the model summary to
            terminal_logger (Terminal_Logger): custom class to save terminal output to a log file
        Returns:
            None
        """
        # save the model summary to a file in the model_and_metrics_directory by redirecting the terminal output to a file
        summary_save_path = self.model_and_metrics_dir.joinpath("model_summary.txt")
        with open(summary_save_path, "w") as sys.stdout:
            # write to file the input shape used in creating the model
            print(f"Input shape used to build the model: {model.input_shape[1:]}")
            print(f"Number of classes in the classification problem: {num_of_classes}\n")
            model.summary()
        
        # reconnect the terminal output to the log file to continue printing to the terminal and log file
        self.terminal_logger.reconnect_to_log_file()
        
        print(f"model summary saved to {summary_save_path}\n")
        return
    
    def save_model_architecture_to_file(self, model):
        """
        Save an image of the model architecture as a graph to model_graph.png.  
        NOTE: This function will only run on a Linux system and if Graphviz is installed.  I'm not sure if there is an easy way to check if 
        Graphviz is installed on a Windows system, but this should be able to run on a Windows system if Graphviz is installed.
        
        Parameters:
            model: keras model object, the model to generate the graph image of
        Returns:
            None
        """
        # set the img save path to the model_and_metrics_directory
        img_save_path = self.model_and_metrics_dir.joinpath("model_graph.png")
        # Check if running on Linux and if Graphviz is installed on the Linux system
        if platform.system() == 'Linux' and check_graphviz_installed():
            # Visualize the model
            keras.utils.plot_model(model, to_file=img_save_path, show_shapes=True, show_layer_names=True)
            print(f"Model architecture visualized as a graph and saved to: {img_save_path}")
        else:
            print("Skipping visualizing model architecture as a graph.  This function only runs on a Linux system and if Graphviz is installed.")
        return
    
    def print_and_save_train_test_results(self, training_history, test_loss, test_accuracy):
        """
        Function to print and save the results of training and testing to a txt file.  Also plots the training and validation loss to a plot
        and saves it.
        
        Parameters: 
            training_history (keras.callbacks.History): history object from model training
            test_loss (float): loss of the model on the test set, a None value indicates that the model was not evaluated on the test set
            test_accuracy (float): accuracy of the model on the test set, a None value indicates that the model was not evaluated on the test set
        Returns:
            None
        """
        # get the best epoch from the best_model_path
        best_model_filename = self.best_model_path.name
        self.best_epoch = int(best_model_filename.split("-")[1].split("_")[1])
        
        # determine the final training and validation loss from the best epoch (epoch with best val loss)
        self.final_training_loss = training_history.history['loss'][self.best_epoch-1]
        self.final_validation_loss = training_history.history['val_loss'][self.best_epoch-1]
        
        # print and save the results to a txt file
        for print_destination in ["to_screen", "to_file"]:
            # when printing to a file, redirect 'sys.stdout' to the file
            if print_destination == "to_file":
                results_save_path = self.model_and_metrics_dir.joinpath("training_results.txt")
                sys.stdout = open(results_save_path, "w")
            
            # print the results from training and testing (with weird formatting to make it look nice)
            print(f"{'-' * 18} RESULTS {'-' * 18}")
            print(f"Training time:{self.total_train_time_string:>31}")
            print(f"Average time per epoch:{self.time_per_epoch_string:>22}")
            print()
            print(f"Total epochs trained:{len(training_history.history['loss']):>24}")
            print(f"Best model from epoch:{self.best_epoch:>23}")
            print()
            print(f"Training loss from epoch   {self.best_epoch:>4}:{self.final_training_loss:>13.3f}")
            print(f"Validation loss from epoch {self.best_epoch:>4}:{self.final_validation_loss:>13.3f}")
            if test_loss is not None:
                print(f"Test loss: {test_loss:>34.3f}")
            print()
            print(f"Training accuracy:{training_history.history['acc'][self.best_epoch-1]:>27.2%}")
            print(f"Validation accuracy:{training_history.history['val_acc'][self.best_epoch-1]:>25.2%}")
            if test_accuracy is not None:
                print(f"Test accuracy:{test_accuracy:>31.2%}")
            print(f"{'-' * 45}")
            
            if print_destination == "to_file":
                # reconnect the terminal output to the log file
                self.terminal_logger.reconnect_to_log_file()
                print(f"training results saved to {results_save_path}")
        
        # give a final update to the training and validation loss plot
        linear_fig_save_path, log_fig_save_path = plot_training_and_validation_loss(self.csv_filename, self.best_epoch)
        
        # if test set was evaluated and mlflow is used to track information, log the test set metrics to MLFlow
        if self.using_mlflow and test_loss is not None:
            # log the test loss and accuracy to MLFlow
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("test_accuracy", test_accuracy)
        
        # if using mlflow, log some additional metrics and end the run
        if self.using_mlflow:
            mlflow.log_metric("final_val_loss", self.final_validation_loss)
            mlflow.log_metric("final_train_loss", self.final_training_loss)
            mlflow.log_metric("best_epoch", self.best_epoch)
            mlflow.log_metric("total_train_time_seconds", self.total_train_time_seconds)
            mlflow.log_metric("time_per_epoch_seconds", self.time_per_epoch_seconds)
            # log the custom plots for the training and validation loss
            mlflow.log_artifact(linear_fig_save_path)
            mlflow.log_artifact(log_fig_save_path)
            # end the mlflow run
            mlflow.end_run()
        
        # print out where the best model is saved because we are at the end of the script and want this information logged by the terminal logger
        print(f"best model located at: {self.best_model_path}\n")
        return


def jax_gpu_check():
    """
    function that will check if there is a gpu available for JAX to use.  If there is a gpu available, the function will return True, otherwise it 
    will return False.
    
    Parameters:
        None
    Returns:
        gpu_available (bool): True if a gpu is available for JAX to use, False otherwise
    """
    try:
        gpu_devices = jax.devices("gpu")
        if gpu_devices:
            return True
        else:
            return False
    except RuntimeError:
        return False


def tf_gpu_check():
    """
    Function that will check if there is a GPU available for TensorFlow to use. 
    If there is a GPU available, the function will return True, otherwise it will return False.
    
    Parameters:
        None
    Returns:
        gpu_available (bool): True if a GPU is available for TensorFlow to use, False otherwise
    """
    try:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        if gpu_devices:
            return True
        else:
            return False
    except RuntimeError as e:
        print(e)
        return False


def clean_up_model_save_dir(checkpoint_dir: PurePath):
    """
    The checkpoint callback saves a model every time the validation loss improves leading to many models being saved.  This function removes
    all but the best model from the model save directory by deleting all but the last model saved.
    
    Parameters:
        checkpoint_dir (PurePath): path to the directory where the model checkpoints are saved
    Returns:
        None
    """
    # remove all but the last model (the best model)
    for model_save in os.listdir(checkpoint_dir)[:-1]:
        # remove all model checkpoints that were saved on the way to the best model
        intermediate_model_path = checkpoint_dir.joinpath(model_save)
        os.remove(intermediate_model_path)
    return


def load_keras_model(model_path: PurePath) -> keras.Model:
    """
    load a keras model
    
    Parameters:
        model_path (PurePath): path to the model to load
    Returns:
        model (keras.Model): the loaded keras model
    """
    # load the keras model
    model = keras.models.load_model(model_path)
    return model


def check_graphviz_installed():
    """
    Function to check if Graphviz is installed on the Linux system.
    
    Parameters:
        None
    Returns:
        bool: True if Graphviz is installed, False otherwise
    """
    try:
        subprocess.check_call(['dot', '-V'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False


def plot_training_and_validation_loss(csv_filename: PurePath, best_epoch=None):
    """
    Plot the training and validation loss from model training called by the Time_History_Callback to give a visual representation of the training
    progress.  Two versions of the plot are created: one with a linear y-axis scale and one with a log y-axis scale.  The plot will be saved to the
    model_and_metrics_directory (same directory as the csv file).  The final plot is created after training by the train_params_and_info class method
    method print_and_save_train_test_results().  If the best_epoch is provided, typically after training, the plot will have a vertical line to show
    the epoch that produced the best model.
    
    If the csv file only contains a header and no data, then a blank plot will be saved.
    During training when no best_epoch is provided, the best_epoch will be ignored.
    After training when the best_epoch is provided, a vertical line will be added to the plot to show the best_epoch.
    
    Parameters:
        csv_filename (PurePath): path to the csv file that contains the training logs
        best_epoch (int): epoch that produced the best model
    Returns:
        linear_fig_save_path (PurePath): path to the linear scale plot
        log_fig_save_path (PurePath): path to the log scale plot
    """
    # read the csv file into a pandas dataframe
    training_history_df = pd.read_csv(csv_filename)
    
    # define a holder var for the linear save path
    linear_fig_save_path = None
    
    # create two versions of the plot: one with a linear y-axis scale and one with a log y-axis scale
    y_scale_versions = ['linear', 'log']
    for y_scale in y_scale_versions:
        # create plot
        plt.figure(figsize=(10, 6))
        # if the training_history_df only contains one row of data, then plot dots instead of lines
        if len(training_history_df) == 1:
            plt.plot(training_history_df['epoch'], training_history_df['val_loss'], 'o', label='validation loss', zorder=2)
            plt.plot(training_history_df['epoch'], training_history_df['loss'], 'o', label='training loss', zorder=2)
        else: # plot lines
            plt.plot(training_history_df['epoch'], training_history_df['val_loss'], label='validation loss', zorder=2)
            plt.plot(training_history_df['epoch'], training_history_df['loss'], label='training loss', zorder=2)
        
        # set the y-axis scale to log if y_scale is log
        if y_scale == 'log':
            plt.yscale('log')
        
        # if the best_epoch is not provided, then assume we are plotting during training.  Show the current epoch and loss on the plot
        if best_epoch is None:
            # extract the current epoch and loss from the dataframe
            current_epoch = training_history_df['epoch'].iloc[-1]
            current_val_loss = training_history_df['val_loss'].iloc[-1]
            current_train_loss = training_history_df['loss'].iloc[-1]
            
            # determine what the best validation loss and what epoch it occurred at
            best_val_loss = training_history_df['val_loss'].min()
            best_val_loss_epoch = training_history_df.loc[training_history_df['val_loss'] == best_val_loss, 'epoch'].values[0]
            epochs_since_best = current_epoch - best_val_loss_epoch
            
            # add a dot for the best validation loss so far
            plt.plot(best_val_loss_epoch, best_val_loss, 'o', color='black', alpha=0.8, label='best val loss so far', zorder=2)
            
            # create TextAreas for each part of the metrics text
            current_epoch_text = TextArea(f"current epoch: {current_epoch}", textprops=dict(color="black", fontsize=9))
            current_val_loss_text = TextArea(f"current val loss: {current_val_loss:.3f}", textprops=dict(color="#185B8C", fontsize=9))
            current_train_loss_text = TextArea(f"current train loss: {current_train_loss:.3f}", textprops=dict(color="#CC5F00", fontsize=9))
            empty_space_text = TextArea("\n", textprops=dict(color="black", fontsize=4))
            best_epoch_text = TextArea(f"best epoch based on val loss: {best_val_loss_epoch}", textprops=dict(color="black", fontsize=9))
            best_val_loss_text = TextArea(f"best val loss: {best_val_loss:.3f}", textprops=dict(color="#185B8C", fontsize=9))
            epochs_since_best_text = TextArea(f"epochs since best val loss: {epochs_since_best}", textprops=dict(color="black", fontsize=9))
            
            # combine the TextAreas using VPacker (vertical packer)
            text_box = VPacker(children=[
                current_epoch_text, 
                current_val_loss_text, 
                current_train_loss_text, 
                empty_space_text,
                best_epoch_text, 
                best_val_loss_text, 
                epochs_since_best_text], 
                align="right", pad=0, sep=3)
            
            # set the position of the text box to the top of the plot (will be offset by xybox in AnnotationBbox)
            y_top = plt.ylim()[1]
            x_right = plt.xlim()[1]
            
            # create an AnnotationBbox with the combined text box
            annotation_box = AnnotationBbox(text_box, (x_right, y_top),
                                            xybox=(-81, -45),  # offset the box so that the point provided is the top right
                                            xycoords='data',
                                            boxcoords="offset points",
                                            frameon=True,
                                            bboxprops=dict(facecolor='white', edgecolor='#D6D6D6'))
            
            # add the AnnotationBbox to the plot
            plt.gca().add_artist(annotation_box)
        
        else:  # if the best_epoch is provided, add information to the plot for the best_epoch
            # extract the final validation loss and training loss from dataframe using best epoch
            final_validation_loss = training_history_df.loc[training_history_df['epoch'] == best_epoch, 'val_loss'].values[0]
            final_training_loss = training_history_df.loc[training_history_df['epoch'] == best_epoch, 'loss'].values[0]
            
            # add a vertical line for the epoch that produced the best model
            plt.axvline(x=best_epoch, color='black', linestyle='--', label='best model epoch', zorder=1, alpha=0.35)
            
            # create TextAreas for each part of the metrics text
            best_epoch_text = TextArea(f"best model epoch: {best_epoch}", textprops=dict(color="black", fontsize=9))
            best_val_loss_text = TextArea(f"best val loss from epoch {best_epoch}: {final_validation_loss:.3f}", textprops=dict(color="#185B8C", fontsize=9))
            train_loss_text = TextArea(f"train loss at epoch {best_epoch}: {final_training_loss:.3f}", textprops=dict(color="#CC5F00", fontsize=9))
            
            # combine the TextAreas using VPacker (vertical packer)
            text_box = VPacker(children=[best_epoch_text, best_val_loss_text, train_loss_text], align="right", pad=0, sep=3)
            
            # set the position of the text box to the top of the plot (will be offset by xybox in AnnotationBbox)
            y_top = plt.ylim()[1]
            
            # create an AnnotationBbox with the combined text box
            annotation_box = AnnotationBbox(text_box, (best_epoch, y_top),
                                            xybox=(-86, -23),  # offset the box so that the point provided is the top right
                                            xycoords='data',
                                            boxcoords="offset points",
                                            frameon=True,
                                            bboxprops=dict(facecolor='white', edgecolor='#D6D6D6'))
            
            # add the AnnotationBbox to the plot
            plt.gca().add_artist(annotation_box)
        
        # add a grid to the plot
        plt.grid(alpha=0.35, zorder=0)
        
        # format the x-axis ticks based on the number of epochs
        epoch_list = training_history_df['epoch']
        # if the number of epochs is less than 50, show every epoch on the x-axis
        if len(epoch_list) < 50:
            plt.xticks(epoch_list)
        # if the number of epochs is greater than 50 but less than 300, show every 5th epoch on the x-axis
        elif len(epoch_list) < 300:
            xtick_list = [1]
            # increment by 5 starting from 5
            xtick_list.extend(list(range(5, len(epoch_list) + 1, 5)))
            # if the last epoch is not a multiple of 5, then add the next multiple of 5
            if xtick_list[-1] != len(epoch_list):
                xtick_list.append(xtick_list[-1] + 5)
            plt.xticks(xtick_list)
        else: # the number of epochs is greater than 300, show every 50th epoch on the x-axis
            xtick_list = [1]
            # increment by 50 starting from 50
            xtick_list.extend(list(range(50, len(epoch_list) + 1, 50)))
            # if the last epoch is not a multiple of 50, then add the next multiple of 50
            if xtick_list[-1] != len(epoch_list):
                xtick_list.append(xtick_list[-1] + 50)
            plt.xticks(xtick_list)
        # Rotate the x-axis labels for better readability
        plt.xticks(rotation=90)
        
        plt.xlabel('Epochs')
        if y_scale == 'linear':
            plt.ylabel('Loss')
        else:
            plt.ylabel('Loss (log scale)')
        plt.title('Training and validation loss')
        plt.legend(loc='lower left', fontsize=6)
        plt.tight_layout()
        
        # save the plot to the model_and_metrics_directory (same directory as the csv file)
        model_and_metrics_directory = csv_filename.parent
        if y_scale == 'linear':
            fig_save_path = model_and_metrics_directory.joinpath("training_and_validation_loss.png")
            linear_fig_save_path = fig_save_path
        else:  # y_scale is log
            fig_save_path = model_and_metrics_directory.joinpath("training_and_validation_loss_log_scale.png")
            log_fig_save_path = fig_save_path
        plt.savefig(fig_save_path, dpi=200)
        plt.close()
    
    # only print out that the plot was saved when the best_epoch is provided
    if best_epoch is not None:
        print(f"Training and validation loss plot saved to {linear_fig_save_path}")
    return linear_fig_save_path, log_fig_save_path


def create_placeholder_training_and_validation_loss_plot(csv_filename: PurePath):
    """
    Function to create a placeholder plot for the training and validation loss plot.  It also prints out where users can monitor this
    plot as it is populated with data during training.
    
    Parameters:
        csv_filename (PurePath): path to the csv file that contains the training logs (currently empty)
    Returns:
        None
    """
    
    # create plot
    plt.figure(figsize=(10, 6))
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')
    plt.tight_layout()
    
    # save the plot to the model_and_metrics_directory (same directory as the csv file)
    model_and_metrics_directory = csv_filename.parent
    fig_save_path = model_and_metrics_directory.joinpath("training_and_validation_loss.png")
    plt.savefig(fig_save_path, dpi=200)
    # create a log scale placeholder plot as well
    plt.ylabel('Loss (log scale)')
    fig_save_path = model_and_metrics_directory.joinpath("training_and_validation_loss_log_scale.png")
    plt.savefig(fig_save_path, dpi=200)
    plt.close()
    
    print(f"Placeholder training and validation loss plot saved to {fig_save_path}.")
    print(f"The plot is currently empty, but will be updated as training progresses.\n")
    return
