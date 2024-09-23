"""
Module that holds the main functionality of running an lightly-guided random grid search of hyperparameters for training a ML model. Starting
from a grid search parameter YAML file, this module will generate a grid search space CSV file that holds the hyperparameters for each run as
well as the training metrics and information for each run.  Each time a new set of hyperparameters is queued, a lock file is created to 
prevent other processes from accessing the grid search space CSV file until the parameters have been selected.  This is generally a very quick
process.  Additionally, the selection process will attempt to find a set of hyperparameters that are more than one value different from any
already completed runs.  This is intended to nudge the search in a more evenly distributed direction.

To visualize the grid search space, open up the MLFlow UI and generate a parallel coordinates plot to what influence different combinations of 
hyperparameters have on the training metrics.

NOTE: The code is a little messy and could use some refactoring.  Everything works, but it could be cleaned up a bit.
"""


import os
import random
import time
import sys
import csv
import yaml
from pathlib import PurePath, Path
import pandas as pd
from sklearn.model_selection import ParameterGrid

# add the parent directory of this file to the sys.path so that utils can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils_files.utils import get_time_stamp, convert_readable_time_str_to_seconds, convert_seconds_to_readable_time_str
from utils_files.model_training_utils import Training_Parameters_and_Info

# PATHS to grid search files (this file must be run this files directory)
grid_search_dir = PurePath("grid_search_files")
grid_search_parameters_YAML = grid_search_dir.joinpath("grid_search_parameters.yaml")
grid_search_space_csv = grid_search_dir.joinpath("parameter_grid.csv")
grid_search_status_txt = grid_search_dir.joinpath("parameter_grid_search_status.txt")
grid_search_lock_file = grid_search_dir.joinpath("parameter_grid_lock.lock")

# GLOBALS for grid search
RUN_METRICS_AND_INFO_COLUMN_NAMES = [
    "train_loss", "val_loss", "final_epoch", "train_time", "avg_time_per_epoch", "run_start_time", "status", "run_name"]
DEFAULT_STATUS = "not started"


def set_lock_to_access_grid_search_space(force_update=False):
    """
    Check if the grid search space CSV file is being used by another process. If a lock file exists, wait until it is no longer being used. Once 
    available, create a lock file to prevent other processes from accessing the file. Delete the lock file when done.
    
    NOTE: This code attempts to handle any race conditions that may occur, but could still error out.  It is good first attempt, but a new version
        could be created that properly takes into account the creation time of the lock file.
    
    Parameters:
        force_update (bool): If True, forcefully create a new lock file after 15 seconds of waiting.
    
    Returns:
        None
    """
    start_time = time.time()
    
    # check if the lock file was created more than 15 seconds ago, if so, then assume that the last process to run errored out.  Delete the lock
    # so that this process can continue without waiting.
    if os.path.exists(grid_search_lock_file):
        lock_file_creation_time = os.path.getctime(grid_search_lock_file)
        if (time.time() - lock_file_creation_time) > 15:
            print("lock file was created more than 15 seconds ago, deleting it and creating a new one...")
            os.remove(grid_search_lock_file)
    
    while True:
        try:
            # Attempt to create the lock file atomically
            fd = os.open(grid_search_lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, 'w') as file:
                file.write("lock file to prevent other processes from accessing the grid search space csv file")
            break  # Successfully created lock file, exit loop
        except FileExistsError:
            # If the lock file exists, wait for 1 second and then check again
            time.sleep(1)
        
        # Print status messages based on wait time
        elapsed_time = time.time() - start_time
        if 1 < elapsed_time < 2:
            print("parameter grid csv file is currently in use, waiting for it to be free...")
        elif elapsed_time > 2:
            print(".")
        
        # Error out if wait time exceeds 15 seconds
        if elapsed_time > 15:
            if force_update:
                print("Force update enabled. Deleting existing lock file and creating a new one.")
                try:
                    os.remove(grid_search_lock_file)
                except FileNotFoundError:
                    pass  # The lock file might have been removed by another process
                time.sleep(random.uniform(0.1, 0.5))  # Small random delay before retrying
                continue
            else:
                error_message = (
                    f"ERROR: the grid search space csv file was locked for longer than 15 seconds as noted by the existence of the lock file: "
                    f"{grid_search_lock_file}. Consider deleting this file manually so that the grid search can continue."
                )
                raise FileExistsError(error_message)
    return


def release_lock_on_grid_search_space():
    """
    Release the lock on the grid search space CSV file so that other processes can access it.
    
    Parameters:
        None
    
    Returns:
        None
    """
    try:
        os.remove(grid_search_lock_file)
    except OSError as e:
        print(f"Error releasing lock file: {e}")
    return


def create_parameter_grid():
    """
    Create a parameter grid from the parameters in the grid_search_parameters.yaml file.
    
    Parameters:
        None
    Returns:
        None
    """
    # load the grid search space parameters from the yaml file
    print(f"\nNo parameter grid search space csv file found. Creating a new one...")
    print(f"loading grid search parameters from: {grid_search_parameters_YAML}")
    with open(grid_search_parameters_YAML, 'r') as file:
        grid_search_parameters = yaml.safe_load(file)
    
    # Generate parameter combinations
    grid = ParameterGrid(grid_search_parameters)
    # get column names
    column_names = list(grid.param_grid[0].keys())
    
    # add metrics and info to column names that will be populated when the model is trained
    column_names = column_names + RUN_METRICS_AND_INFO_COLUMN_NAMES
    print(f"Parameter grid generated with {len(grid)} combinations.")
    
    # Write to CSV file
    with open(grid_search_space_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=column_names)
        writer.writeheader()
        
        for params in grid:
            # add empty values for the metrics that will be populated when the model is trained
            for metric in RUN_METRICS_AND_INFO_COLUMN_NAMES:
                if metric == "status":
                    params[metric] = DEFAULT_STATUS
                else:
                    params[metric] = None
            writer.writerow(params)
    print(f"Parameter grid written to: {grid_search_space_csv}\n")
    return


def get_grid_search_status_report(grid_df: pd.DataFrame, print_to_terminal=True):
    """
    Get a report on the current status of the grid search and save it to a text file named parameter_grid_search_status.txt
    
    Parameters:
        grid_df (pandas.DataFrame): The grid dataframe
        print_to_terminal (bool): If True, print the status report to the screen and save to file. If False, only print the status report to file.
    Returns:
        remaining_runs (int): The number of remaining runs in the grid search space
    """
    # filter the grid_df to only make status report using rows that are not retired
    # NOTE: There may be some rows that have retired combinations that are included in this report because they were already
        # completed before the combination was retired.
    non_retired_grid_df = grid_df[grid_df["status"] != "retired"]
    
    # generate a report on the status of the grid search
    search_report_time_stamp = get_time_stamp()
    total_number_of_runs = len(non_retired_grid_df)
    completed_runs = non_retired_grid_df[non_retired_grid_df["status"] == "completed"]
    runs_in_progress = non_retired_grid_df[non_retired_grid_df["status"] == "in progress"]
    not_started_runs = non_retired_grid_df[non_retired_grid_df["status"] == DEFAULT_STATUS]
    percentage_completed = len(completed_runs) / total_number_of_runs * 100
    average_train_time_string = "unknown"
    estimated_time_to_completion = "unknown"
    
    # load in the current search space parameter options from the yaml file
    with open(grid_search_parameters_YAML, 'r') as file:
        grid_search_parameters = yaml.safe_load(file)
    
    # calculate the retired parameter options
    retired_parameter_options = {}
    retired_parameter_grid_df = grid_df[grid_df["status"] == "retired"]
    if len(retired_parameter_grid_df) > 0:
        # get the possibly retired parameter options from the retired rows
        possible_retired_parameter_options = {}
        for parameter in grid_search_parameters.keys():
            possible_retired_parameter_options[parameter] = list(retired_parameter_grid_df[parameter].unique())
        
        # check if these parameter options are actually retired
        for parameter, options_list in possible_retired_parameter_options.items():
            for option in options_list:
                # check if this parameter and option are in the current grid_search_parameters
                if option not in grid_search_parameters[parameter]:
                    # add it to the retired_parameter_options dictionary
                    if parameter not in retired_parameter_options:
                        retired_parameter_options[parameter] = []
                    retired_parameter_options[parameter].append(option)
    
    # if there are any completed runs, find the information on the best model so far
    completed_runs = grid_df[grid_df["status"] == "completed"].copy()
    if (len(completed_runs) > 0):
        # get the index of the best model so far based on the lowest validation loss
        best_model_index = completed_runs["val_loss"].idxmin()
        # get the metrics and info for the best model so far
        best_model_train_loss = grid_df.loc[best_model_index, "train_loss"]
        best_model_val_loss = grid_df.loc[best_model_index, "val_loss"]
        best_model_final_epoch = grid_df.loc[best_model_index, "final_epoch"]
        best_model_train_time = grid_df.loc[best_model_index, "train_time"]
        best_model_avg_time_per_epoch = grid_df.loc[best_model_index, "avg_time_per_epoch"]
        best_model_run_name = grid_df.loc[best_model_index, "run_name"]
        
        # get the parameters that were used to train the best model so far
        best_model_parameters = grid_df.loc[best_model_index, grid_search_parameters.keys()]
        
        # convert the train time from a readable time string to seconds
        completed_runs['train_time'] = completed_runs['train_time'].apply(convert_readable_time_str_to_seconds)
        # calculate the average time in seconds for all completed runs
        average_train_time_seconds = completed_runs["train_time"].mean()
        # format as a readable time string
        average_train_time_string = convert_seconds_to_readable_time_str(average_train_time_seconds)
        # estimate the estimated_time_to_completion based on the average time per epoch and the number of not started runs
        if len(not_started_runs) > 0:
            estimated_time_to_completion_seconds = average_train_time_seconds * len(not_started_runs)
            estimated_time_to_completion = convert_seconds_to_readable_time_str(estimated_time_to_completion_seconds)
        # if we are out of not started runs, set the estimated time to completion to 0 seconds (since the grid search is complete)
        elif len(not_started_runs) == 0:
            estimated_time_to_completion = convert_seconds_to_readable_time_str(seconds=0)
    
    # if there are retired parameters, print out each retired parameter
    for print_destination in ["to_terminal", "to_file"]:
        # if not printing to the screen, skip the print to screen section
        if not print_to_terminal:
            continue
        
        # when printing to a file, redirect 'sys.stdout' to the file
        if print_destination == "to_file":
            sys.stdout = open(grid_search_status_txt, "w")
        
        print(f"{'-' * 22} GRID SEARCH STATUS {'-' * 22}")
        print(f"Gird search status report generated at:  {search_report_time_stamp}")
        print()
        print(f"Total number of runs: {total_number_of_runs}")
        print(f"Completed runs: {len(completed_runs)}")
        print(f"Runs in progress: {len(runs_in_progress)}")
        print(f"Not started runs: {len(not_started_runs)}")
        print(f"Percentage of runs completed: {percentage_completed:.2f}%")
        print(f"Average time to train a model: {average_train_time_string}")
        print(f"Estimated time to complete grid search: {estimated_time_to_completion}")
        print()
        
        # if there are any completed runs, print out the best model so far
        if (len(completed_runs) > 0):
            print(f"BEST MODEL SO FAR:")
            print(f"    Validation loss: {best_model_val_loss:.3f}")
            print(f"    Train loss: {best_model_train_loss:.3f}")
            print(f"    Final epoch: {best_model_final_epoch}")
            print(f"    Train time: {best_model_train_time}")
            print(f"    Avg time per epoch: {best_model_avg_time_per_epoch}")
            print(f"    Run name: {best_model_run_name}")
            print()
            print("BEST MODEL PARAMETERS:")
            for parameter, option in best_model_parameters.items():
                print(f"    {parameter}: {option}")
            print
        
        # if there are retired parameter options, print out the parameter its options that have been retired
        if len(retired_parameter_options) > 0:
            print(f"RETIRED PARAMETER OPTIONS:")
            for parameter, options in retired_parameter_options.items():
                print(f"{parameter}: {options}")
        print(f"{'-' * 64}")
        
        if print_destination == "to_file":
            # reconnect the terminal output to the log file
            sys.stdout = sys.__stdout__
            print(f"grid search status saved to {grid_search_status_txt}\n")
    
    # format the runs not started into an integer by counting the number of rows in the not_started_runs dataframe
    remaining_runs = len(not_started_runs)
    return remaining_runs


def reset_stale_in_progress_runs(grid_df: pd.DataFrame):
    """
    Resets the status and training metrics of rows in the grid dataframe that have been in progress for more than 24 hours.
    
    Parameters:
        grid_df (pandas.DataFrame): The grid dataframe
    
    Returns:
        pandas.DataFrame: The updated grid dataframe with the status and training metrics reset for stale in-progress runs.
    """
    # for every row that is "in progress", but the run started more than 24 hours ago, reset the status to "not started" and update the
    # columns that are populated with the training metrics to "None"
    in_progress_rows = grid_df[grid_df["status"] == "in progress"]
    for index, row in in_progress_rows.iterrows():
        # get the rows start time as a pandas datetime object
        run_start_time = row["run_start_time"]
        run_start_time = pd.to_datetime(run_start_time, format="%Y_%m_%d__%H_%M_%S_%f")
        # if the run started more than 24 hours ago...
        if (pd.Timestamp.now() - run_start_time).total_seconds() > 86400:
            print(f"\nSTALE RUN DETECTED...")
            print(f"run with index {index} has been in progress for longer than 24 hours.")
            print(f"Resetting the row for future grid searches\n")
            # reset all of the training metrics and status for this row
            for metric in RUN_METRICS_AND_INFO_COLUMN_NAMES:
                if metric == "status":
                    grid_df.loc[index, metric] = DEFAULT_STATUS
                else:
                    grid_df.loc[index, metric] = None
    return grid_df


def updated_grid_with_latest_grid_search_parameters(existing_grid_df: pd.DataFrame):
    """
    Check the grid search parameters yaml file to see if any hyperparameters options have been added or removed.
    As I search the grid for the best hyperparameters, I may decided I want to modify the grid_search_parameters.yaml file to add or remove
    hyperparameters.  This function will update the grid with the latest parameters from the yaml file.
    
    NOTE: this will only work for changing the hyperparameters options, not adding new hyperparameters to the grid.
    example: 
        original early_stopping_patience: [10, 20, 30]
        updated early_stopping_patience: [10, 20]
    
    Parameters:
        existing_grid_df (pandas.DataFrame): The grid dataframe
    Returns:
        existing_grid_df (pandas.DataFrame): The updated grid dataframe with the latest hyperparameters options from the yaml file.
    """
    ###### STEP 1: Generate a new dataframe with the latest grid search parameters and default values for the metrics and info columns
    # load the grid search space parameters from the yaml file
    with open(grid_search_parameters_YAML, 'r') as file:
        latest_grid_search_parameters = yaml.safe_load(file)
    parameter_names = list(latest_grid_search_parameters.keys())
    # format the latest grid search parameters as a pandas dataframe
    latest_grid_df = pd.DataFrame(ParameterGrid(latest_grid_search_parameters))
    # add the default values for the metrics and info columns to the latest_grid_combinations_df
    for metric in RUN_METRICS_AND_INFO_COLUMN_NAMES:
        if metric == "status":
            # add a new column with the status set to "not started" for all rows
            latest_grid_df["status"] = DEFAULT_STATUS
        else:
            # add a new column with the metric set to None for all rows
            latest_grid_df[metric] = None
    
    ###### STEP 2: extract the unique values from the existing grid's parameter columns
    # filter the grid to only look at parameter columns
    filtered_existing_grid_df = existing_grid_df[parameter_names]
    existing_grid_search_parameters = {}
    # get the existing_grid_search_parameters from unique values in the grid_df
    for parameter in parameter_names:
        # get the unique values for this parameter
        unique_values = filtered_existing_grid_df[parameter].unique()
        # add this parameter to the existing_grid_search_parameters dictionary
        existing_grid_search_parameters[parameter] = list(unique_values)
    # raise value error if the keys from each dictionary are not the same
    if existing_grid_search_parameters.keys() != latest_grid_search_parameters.keys():
        raise ValueError("the keys from the grid_search_parameters.yaml file no longer match the parameter columns in the parameter_grid.csv file.")
    
    ###### STEP 3: check if any of the parameters have new or removed options
    for parameter in parameter_names:
        # extract the existing and latest parameter options
        existing_parameter_options = existing_grid_search_parameters[parameter]
        latest_parameter_options = latest_grid_search_parameters[parameter]
        # sort the two option lists for comparison
        existing_parameter_options.sort()
        latest_parameter_options.sort()
        # if the parameter options are different, update the grid_df with the latest options
        if existing_parameter_options != latest_parameter_options:
            # check to see if any of the latest options are not in the existing options (new options added to the yaml file)
            for latest_option in latest_parameter_options:
                if latest_option not in existing_parameter_options:
                    print(f"New parameter option detected: {parameter}: {latest_option}")
                    # extract all rows from the latest_grid_df that use this new option
                    new_rows = latest_grid_df[latest_grid_df[parameter] == latest_option]
                    # add these rows to the end of the existing grid_df
                    existing_grid_df = pd.concat([existing_grid_df, new_rows], ignore_index=True)
            # check to see if any of the existing options are not in the latest options (options removed from the yaml file)
            for existing_option in existing_parameter_options:
                if existing_option not in latest_parameter_options:
                    print(f"parameter options retired: {parameter}: {existing_option}")
                    print(f"setting the status of unused parameter combinations with {parameter}={existing_option} to 'retired'\n")
                    # extract all rows from the existing_grid_df that use this retired option
                    retired_rows = existing_grid_df[existing_grid_df[parameter] == existing_option]
                    # further filter the retired_rows to only include rows that are "not started"
                    retired_rows = retired_rows[retired_rows["status"] == DEFAULT_STATUS]
                    # update the status of these rows in the existing_grid_df to "retired"
                    existing_grid_df.loc[retired_rows.index, "status"] = "retired"
    
    # ###### STEP 4: un-retire any retired rows that have been re-added to the grid search space
    # filter the grid to only look at retired rows
    retired_rows = existing_grid_df[existing_grid_df["status"] == "retired"]
    # create params only copies of the latest_grid_df and retired_rows for better comparison
    params_only_latest_grid_df = latest_grid_df.drop(columns=RUN_METRICS_AND_INFO_COLUMN_NAMES)
    params_only_retired_rows = retired_rows.drop(columns=RUN_METRICS_AND_INFO_COLUMN_NAMES)
    
    # if any of the rows in retired_rows are in the latest_grid_df, un-retire them from the existing_grid_df
    for index, row in params_only_retired_rows.iterrows():
        # format the row as a dictionary
        retired_row_dict = row.to_dict()
        # see if the row exists in the latest_grid_df
        possible_row_index = params_only_latest_grid_df[params_only_latest_grid_df.eq(retired_row_dict).all(1)].index.tolist()
        # if possible row index is not empty, then it exists in the latest_grid_df
        if len(possible_row_index) > 0:
            # use the index from retired_rows to un-retire the row in the existing_grid_df
            existing_grid_df.loc[index, "status"] = DEFAULT_STATUS
    
    return existing_grid_df


def find_diverse_random_train_parameters_from_grid(grid_df: pd.DataFrame):
    """
    helper function used by get_random_train_parameters_from_grid which attempts to find a set of training parameters that are more than one
    value different from any already completed runs.  This process is intended to nudge the search in a more evenly distributed direction.
    
    Example:
    if run 1 has parameters: {"a_param: 1, "b_param": 1, "c_param": 1}"}
    Then we wouldn't want to select a set of parameters where only one of the values is different from run 1. Meaning, at least two of the
    parameters should be different.
    non-acceptable run 2: {"a_param: 1, "b_param": 2, "c_param": 1}"}
    acceptable run 2: {"a_param: 1, "b_param": 2, "c_param": 2}"}
    
    Parameters:
        grid_df (pandas.DataFrame): The grid dataframe
    Returns:
        grid_df (pandas.DataFrame): The updated grid dataframe with the status of the selected row set to "in progress"
        training_parameters (dict): A dictionary of training parameters
    """
    # recreate the grid search space as a dictionary
    parameters_only_grid_df = grid_df.drop(columns=RUN_METRICS_AND_INFO_COLUMN_NAMES)
    grid_search_parameters = {}
    for column in parameters_only_grid_df.columns:
        grid_search_parameters[column] = list(parameters_only_grid_df[column].unique())
    
    # try to find a random row that is sufficiently diverse from the completed runs, if we can't find one after 40 tries, then we assume that
    # the search space is running out of diverse options and we will accept the last random row to continue the search.
    diverse_random_row_check_count = 0
    while diverse_random_row_check_count < 40:
        # get a random row from the dataframe where the status is "not started"
        random_row = grid_df[grid_df["status"] == DEFAULT_STATUS].sample()
        # assume the random row is sufficiently diverse and check to see if it is not
        sufficiently_diverse_random_row = True
        
        # for parameter in random_row, check to see if there are any completed runs that are only one value different from this row and if so,
        # get a new random row
        for parameter, option_list in grid_search_parameters.items():
            # get the parameter value from the random row
            random_row_parameter_value = random_row[parameter].values[0]
            for new_parameter_option in option_list:
                if new_parameter_option != random_row_parameter_value:
                    # create a new row with this parameter option
                    off_by_one_row = random_row.copy()
                    off_by_one_row[parameter] = new_parameter_option
                    # remove the columns that are not training parameters from the row
                    off_by_one_row = off_by_one_row.drop(columns=RUN_METRICS_AND_INFO_COLUMN_NAMES)
                    # format the off_by_one_row into a dictionary
                    off_by_one_random_row_dict = off_by_one_row.to_dict(orient="records")[0]
                    # find the index of this off_by_one_row in the parameters_only_grid_df
                    off_by_one_row_index = parameters_only_grid_df[parameters_only_grid_df.eq(off_by_one_random_row_dict).all(axis=1)].index[0]
                    # check the status of this row in the grid_df to see if it is "completed" or "in progress"
                    off_by_one_row_status = grid_df.loc[off_by_one_row_index]["status"]
                    if off_by_one_row_status == "completed" or off_by_one_row_status == DEFAULT_STATUS:
                        # mark the random row as not sufficiently diverse
                        sufficiently_diverse_random_row = False
                        break
            # if the random row is not sufficiently diverse, break the loop and get a new random row
            if not sufficiently_diverse_random_row:
                break
        # if the random row is sufficiently diverse, break the loop and use this row
        if sufficiently_diverse_random_row:
            break
        # increment the diverse_random_row_check_count
        diverse_random_row_check_count += 1
    
    # update this row in grid_df to show that it is in progress
    grid_df.loc[random_row.index, "status"] = "in progress"
    # set the run start time for this row in grid_df (and use the data type of string)
    grid_df.loc[random_row.index, "run_start_time"] = get_time_stamp()
    
    # remove the columns that are not training parameters from the row
    random_row = random_row.drop(columns=RUN_METRICS_AND_INFO_COLUMN_NAMES)
    
    # convert the random_row to a dictionary of training parameters
    training_parameters = random_row.to_dict(orient="records")[0]
    
    return grid_df, training_parameters


def set_default_column_dtypes_to_avoid_errors(grid_df: pd.DataFrame):
    """
    Sets default column data types to avoid errors.  Probably only necessary until an all NaN column is populated with a string value.
    
    This function takes a pandas DataFrame as input and sets the data types of specific columns to string. 
    It is designed to avoid data type errors that may occur when working with columns that currently have all NaN values.
    
    Parameters:
    - grid_df (pandas DataFrame): The input DataFrame to modify.
    
    Returns:
    - grid_df (pandas DataFrame): The modified DataFrame with updated column data types.
    """
    # to avoid data type errors on first run, set string data types for columns that currently have all NaN values
    grid_df["train_time"] = grid_df["train_time"].astype(str)
    grid_df["avg_time_per_epoch"] = grid_df["avg_time_per_epoch"].astype(str)
    grid_df["run_start_time"] = grid_df["run_start_time"].astype(str)
    grid_df["status"] = grid_df["status"].astype(str)
    grid_df["run_name"] = grid_df["run_name"].astype(str)
    return grid_df


def get_random_train_parameters_from_grid():
    """
    Steps this function takes to get a random set of training parameters from the grid search space:
    1. lock the grid search space csv file so that other processes can't access it.  Wait if other processes are using it.
    2. read in the grid search space csv file
    3. reset the status of any runs that have been in progress for more than 24 hours to "not started"
    4. update the grid if there have been any changes to the grid search parameters yaml file
    5. attempt to find a diverse random row from the grid search space for this training run
    6. update the status of this row in grid_df to in progress with a start time of the current time
    7. save the updated dataframe back to the csv file
    8. update or create a grid search status report
    9. return a dictionary of training parameters (excluding the training metrics and info columns)
    10. release the lock on the grid search space csv file so that other processes can access it
    
    Parameters:
        None
    Returns:
        training_parameters (dict): A dictionary of training parameters, empty if there are no runs left in the grid search space
        remaining_runs (int): The number of remaining runs in the grid search space
    """
    # set the lock to access the grid search space csv file
    set_lock_to_access_grid_search_space()
    
    # if the parameter grid file does not exist, create it
    if not os.path.exists(grid_search_space_csv):
        create_parameter_grid()
    
    # load the csv file as a pandas dataframe
    grid_df = pd.read_csv(grid_search_space_csv)
    grid_df = set_default_column_dtypes_to_avoid_errors(grid_df)
    
    # reset the status of any runs that have been in progress for more than 24 hours
    grid_df = reset_stale_in_progress_runs(grid_df)
    
    # update the grid if there have been any changes to the grid search parameters yaml file
    grid_df = updated_grid_with_latest_grid_search_parameters(grid_df)
    
    # update the grid search status report and get the number of remaining runs
    remaining_runs = get_grid_search_status_report(grid_df, print_to_terminal=False)
    
    # if there are runs left in the grid search space, continue with the search
    if remaining_runs != 0:
        # attempt to find a diverse random row from the grid search space for this training run. Update the status of this row in grid_df to
        # "in progress" and update the run_start_time to the current time.
        grid_df, training_parameters = find_diverse_random_train_parameters_from_grid(grid_df)
    else: # if there are no runs left in the grid search space, return an empty dictionary
        training_parameters = {}
    
    # save the updated dataframe back to the csv file
    grid_df.to_csv(grid_search_space_csv, index=False)
    
    # update the grid search status report again to reflect the in-progress run
    remaining_runs = get_grid_search_status_report(grid_df)
    
    # release the lock on the grid search space csv file so that other processes can access it
    release_lock_on_grid_search_space()
    return training_parameters, remaining_runs


def log_results_to_grid_search_space_csv(train_params_and_info: Training_Parameters_and_Info):
    """
    Log the results of a training run to the grid search space csv file.
    
    Parameters:
        train_params_and_info (Training_Parameters_and_Info): A Training_Parameters_and_Info object containing the training metrics and info.
    Returns:
        None
    """
    # package up the grid search results information as a dictionary for logging to the parameter grid csv file
    grid_search_results = {
        "train_loss": train_params_and_info.final_training_loss,
        "val_loss": train_params_and_info.final_validation_loss,
        "final_epoch": train_params_and_info.best_epoch,
        "train_time": train_params_and_info.total_train_time_string,
        "avg_time_per_epoch": train_params_and_info.time_per_epoch_string,
        "status": "completed",
        "run_name": train_params_and_info.run_name,
    }
    
    # set the lock to access the grid search space csv file using force_update to ensure that the information from this run is logged
    set_lock_to_access_grid_search_space(force_update=True)
    
    # load the gird search space as a pandas dataframe
    grid_df = pd.read_csv(grid_search_space_csv)
    grid_df = set_default_column_dtypes_to_avoid_errors(grid_df)
    
    # create a copy of grid_df that only has the parameter columns
    parameters_only_grid_df = grid_df.drop(columns=RUN_METRICS_AND_INFO_COLUMN_NAMES)
    
    # convert the training_parameters to a dictionary
    train_params_and_info = train_params_and_info.get_training_parameters_dict()
    
    # find the row in the grid_df that matches the training_parameters
    row_index = parameters_only_grid_df[parameters_only_grid_df.eq(train_params_and_info).all(axis=1)].index[0]
    
    # update the row in grid_df with the training metrics
    for metric, value in grid_search_results.items():
        grid_df.loc[row_index, metric] = value
    
    # save the updated dataframe back to the csv file
    grid_df.to_csv(grid_search_space_csv, index=False)
    
    # update the grid search status report
    get_grid_search_status_report(grid_df, print_to_terminal=False)
    
    # release the lock on the grid search space csv file so that other processes can access it
    release_lock_on_grid_search_space()
    
    return


def final_update_grid_search_status_report():
    """
    Final update of the grid search status report after all runs have been completed.  Typically only called when the random grid search is complete.
    It loads the grid search space csv file and updates the status report.
    
    Parameters:
        None
    Returns:
        None
    """
    # set the lock to access the grid search space csv file
    set_lock_to_access_grid_search_space()
    
    # load the csv file as a pandas dataframe
    grid_df = pd.read_csv(grid_search_space_csv)
    grid_df = set_default_column_dtypes_to_avoid_errors(grid_df)
    
    # update the grid search status report
    get_grid_search_status_report(grid_df)
    
    # release the lock on the grid search space csv file so that other processes can access it
    release_lock_on_grid_search_space()
    return


if __name__ == "__main__":
    # if this file is run instead of grid_search.py, then update the grid search space csv file and generate a new search status report
    # if the parameter grid file does not exist, create it
    if not os.path.exists(grid_search_space_csv):
        create_parameter_grid()
    
    # set the lock to access the grid search space csv file
    set_lock_to_access_grid_search_space()
    
    # load the grid search space as a pandas dataframe
    grid_df = pd.read_csv(grid_search_space_csv)
    grid_df = set_default_column_dtypes_to_avoid_errors(grid_df)
    
    # reset the status of any runs that have been in progress for more than 24 hours
    grid_df = reset_stale_in_progress_runs(grid_df)
    
    # update the grid if there have been any changes to the grid search parameters yaml file
    grid_df = updated_grid_with_latest_grid_search_parameters(grid_df)
    
    # save the updated dataframe back to the csv file
    grid_df.to_csv(grid_search_space_csv, index=False)
    
    # update the grid search status report
    get_grid_search_status_report(grid_df)
    
    # release the lock on the grid search space csv file so that other processes can access it.  We are done updating the csv file.
    release_lock_on_grid_search_space()
