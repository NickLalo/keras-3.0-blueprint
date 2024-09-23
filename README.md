# Keras Blueprint for Image Classification

This repo contains a blueprint for image classification using Keras 3.0. The blueprint is designed to be a starting point for supervised image classification 
tasks and can easily be modified to train a regression model or other basic supervised learning tasks.

## Running the Blueprint

1. create an the keras-env conda environment: Create a conda environment using the environment yaml files provided in the _configs directory.  GPU 
and CPU versions of the environment are provided and have been tested on Windows 10 and Windows Subsystem for Linux (WSL2) version: 2.2.4.0.

2. train a model: Run the train_model.py script to train a model.  Information about the model, training parameters, data, and other useful information
are logged to the terminal.  Reading the output of the script will help you understand how the training process progresses, what metrics can be monitored,
and where the information related to the model and training process is stored.

3. grid search: Run the grid_search.py script to perform a grid search over the hyperparameter search space of this model.  The grid search is defined
in grid_search_files/grid_search_parameters.yaml with some default values set.  These options can be added or removed during the grid search and the
script will update the search space accordingly.  New hyperparameters can be added, but require modifications to other files where the new hyperparameter
is used.  When adding a new hyperparameter to the search space the grid search should be reset.  This can be done by deleting the following files:
    - grid_search_files/parameter_grid_search_status.txt
    - grid_search_files/parameter_grid.csv

## Adapting the Blueprint

1. Input Data: This blueprint downloads the MNIST dataset from the Keras library. The MNIST dataset is a collection of 28x28 pixel images of handwritten 
digits.  To use this blueprint with your own data, you will need to modify add data to the "training_data" directory, replace the load_and_process_MNIST_data
function in the train_model.py script with a function that loads and processes your data.  Check the format of the current function to see how this process
should be handled.

2. Model: The model used in this blueprint is defined in the function build_and_compile_model located in the utils_files/model_and_callbacks.py file.
To use this blueprint with your own model, you will need to replace the model defined in this function with your own.

  
## TODOs

- [ ] Add visualization of training progress by logging the gradients to better understand how much the model is changing between epochs
    - [ ] Individual step plot
    - [ ] GIF of all steps
    - [ ] Histogram of all steps
- [ ] Add visualization of training progress by logging the weights to better understand the current state of the model
    - [ ] Individual step plot
    - [ ] GIF of all steps
    - [ ] Histogram of all steps
- [ ] Add visualization of training process by logging the activations of the model to see if there are any dead neurons or other issues
    - [ ] Individual step plot
    - [ ] GIF of all steps
    - [ ] Histogram of all steps
- [ ] Add visualization of the model architecture as a graph saved as a .png file
