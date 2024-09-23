"""
Script to run a random grid search of hyperparameters for training a ML model to classify MNIST digits.

TODO:
    1. write some code to investigate multiprocessing for parallelizing the grid search when running on NERSC.
        1.1. write a stress test to see how many processes can be run in parallel on NERSC using a single node.
            set the random seed and run the same grid search multiple times while varying the number of processes.
            see how many processes can be run in parallel before we start to see a decrease in performance due from the total train time.
        1.2. write a function to start multiple processes in parallel with a hardcoded number of processes found from the stress test.
"""


import numpy as np
from train_model import train_example_MNIST_model
from utils_files.model_training_utils import Training_Parameters_and_Info
from grid_search_files.grid_search_utils import get_random_train_parameters_from_grid, log_results_to_grid_search_space_csv, \
    final_update_grid_search_status_report


if __name__ == "__main__":
    # set random seed for pandas to ensure reproducibility in testing
    np.random.seed(112)
    
    # start up a random grid search for hyperparameters that will go until there are no more runs left in the grid search space
    print(f"starting a random grid search to look for the optimal set of hyperparameters...")
    while True:
        # get a random set of training parameters from the grid search space and the number of runs remaining
        print(f"retrieving random training parameters from the grid search space...")
        random_training_parameters, runs_remaining = get_random_train_parameters_from_grid()
        
        # if we are out of runs in the grid search space, give a final update to the grid search status report and break out of the loop
        if runs_remaining == 0:
            print(f"\n\n{'#'*180}")
            print(f"{' '*67}All runs in the grid search space have been completed.")
            print(f"{'#'*180}\n")
            final_update_grid_search_status_report()
            break
        
        # create a training parameters object to hold information for customizing the training run
        train_params_and_info = Training_Parameters_and_Info(random_training_parameters)
        train_params_and_info.using_grid_search = True
        
        # train the model
        print(f"\n\n{'#'*180}")
        print(f"{' '*67}Training model with parameters from grid search")
        print(f"{'#'*180}\n")
        train_params_and_info = train_example_MNIST_model(train_params_and_info)
        
        # log the results of the grid search
        log_results_to_grid_search_space_csv(train_params_and_info)
