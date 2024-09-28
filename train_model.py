"""
Basic example for training a model using Keras with custom callbacks and logging to create a very informative training experience. This
script can be used as a template for training models in the future.
    
    1. investigate ReduceLROnPlateau callback
    
    2. add some documentation for the parameter search space somewhere
        3.1 a YAML file with the parameter search space for the model
        3.2 a README.md file with the parameter search space and explanation of what each parameter does
        3.3 a csv file containing the grid search and results of runs that have been completed
        3.4 an image of the grid search results in a static format (parallel coordinates plot and scatter plot)
        3.5 a script to interactively view the parallel coordinates plot
"""


print("\nloading libraries to train a deep learning model with Keras 3.0...")
import os
import time

# Note that Keras should only be imported after the backend has been configured. The backend cannot be changed once the package is imported
os.environ["KERAS_BACKEND"] = "tensorflow"  # for now, only jax and tensorflow backends are supported
import keras # even if keras is not directly used in this file, loading here is necessary to set the backend to all Keras operations

from utils_files.model_training_utils import Training_Parameters_and_Info, load_keras_model
from utils_files.load_and_process_MNIST_data import load_and_process_MNIST_data
from utils_files.model_and_callbacks import build_and_compile_model, get_callbacks_for_training
from utils_files.utils import convert_seconds_to_readable_time_str
print("loading libraries complete\n")


def train_example_MNIST_model(train_params_and_info: Training_Parameters_and_Info):
    """
    function to train a basic MNIST model using the parameters passed in.
    
    Parameters:
        train_params_and_info: Training_Parameters object that holds parameters for training the model and other info.  This object will be used to
        setup the various aspects of the training run (model_and_metrics_dir, parameters, model architecture, callbacks, etc.)
    Returns:
        train_params_and_info: Training_Parameters object that holds parameters for training the model and other info.  Can be used by the 
        random_grid_search.py script to log the results of the training run to the grid search csv file.
    """
    # setup the model and metrics directory, terminal_logger, and save the training parameters to a json file
    train_params_and_info.create_model_and_metrics_dir()
    
    # load the MNIST data for training and testing
    x_train, x_val, x_test, y_train, y_val, y_test, num_of_classes, input_shape = load_and_process_MNIST_data(train_params_and_info)
    
    # get the model that will be trained
    model = build_and_compile_model(input_shape, num_of_classes, train_params_and_info)
    
    # get the callbacks for training the model and any necessary variables for later use
    callbacks_list, train_params_and_info = get_callbacks_for_training(train_params_and_info)
    
    # train the model
    training_history = model.fit(
        x_train,
        y_train,
        batch_size=train_params_and_info.batch_size,
        epochs=train_params_and_info.epoch_limit,
        validation_data=(x_val, y_val),
        callbacks=callbacks_list,
    )
    
    # load the best model from training to evaluate on the test set
    model = load_keras_model(train_params_and_info.best_model_path)
    print(f"best model loaded from: {train_params_and_info.best_model_path}\n")
    
    # only evaluate the model on the test set if not part of a grid search
    if not train_params_and_info.using_grid_search:
        print(f"evaluating model on test set...")
        test_loss, test_accuracy = model.evaluate(x_test, y_test)
        print()
    else:  # if part of a grid search, fill in the test loss and accuracy with dummy values
        test_loss, test_accuracy = None, None
    
    # print and save the results of training and testing
    train_params_and_info.print_and_save_train_test_results(training_history, test_loss, test_accuracy)
    
    total_script_time_seconds = time.time() - train_params_and_info.script_start_time
    script_time_string = convert_seconds_to_readable_time_str(total_script_time_seconds)
    print(f"model training script completed in {script_time_string}\n")
    
    # return the results from the training run.  Info is stored in the train_params_and_info object
    return train_params_and_info


if __name__ == "__main__":
    # NOTE: Hardcoded custom parameters for training a model and evaluating on the test set.  These parameters were determined by running the
    #       random_grid_search.py script and finding the best parameters for this model.  Setting them here will allow for the model to be retrained
    #       to validate the results of the grid search ensuring reproducibility and to evaluate the model on the test set.
    # NOTE: The results achieved by the random search are currently not perfectly reproducible even with all of the set random seeds, however, from
    #       a few tests, the results are very close to the original results.
    parameters_dict = {
        "kfold_index": 0,
        "debug_run": False,  # if True, will use a very very minimal dataset to test code execution
        "small_train_set": False,  # if True, will use a small subset of the training data to give a quick test of how the model is training
        "epoch_limit": 999,
        "reduce_lr_on_plateau": True,
        "dropout_rate": 0.25,
        "learning_rate": 0.001,
        "early_stopping_patience": 20,
        "batch_size": 64,
    }
    
    # create a training parameters object to hold information for customizing the training run
    train_params_and_info = Training_Parameters_and_Info(parameters_dict)
    
    # train the model (training metrics and results will be saved to a models_and_metrics directory)
    train_example_MNIST_model(train_params_and_info)
