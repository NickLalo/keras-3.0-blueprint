# script to reset the project's saved data directories to a clean state runnable on Linux or WSL

#!/bin/bash

# Function to prompt the user
prompt_user() {
    read -p "reset all save directories in this project? [Y/n]: " response
    response=${response:-Y}
}

# Function to delete directories and files
delete_directories_and_files() {
    # navigate to the directory where this script is located
    cd "$(dirname "$(readlink -f "$0")")"
    # navigate one level up to the project root directory
    cd ../
    # check if mlflow_data_storage exists
    if [ -d "mlflow_data_storage" ]; then
        rm -rf mlflow_data_storage
        echo "Deleting mlflow_data_storage directory..."
    else
        echo "mlflow_data_storage directory does not exist."
    fi
    
    # check if model_and_metrics_saves exists
    if [ -d "model_and_metrics_saves" ]; then
        rm -rf model_and_metrics_saves
        echo "Deleting model_and_metrics_saves directory..."
    else
        echo "model_and_metrics_saves directory does not exist."
    fi
    
    if [ -f "grid_search_files/parameter_grid.csv" ]; then
        rm "grid_search_files/parameter_grid.csv"
        echo "Deleted grid_search_files/parameter_grid.csv"
    else
        echo "grid_search_files/parameter_grid.csv does not exist."
    fi
    
    if [ -f "grid_search_files/parameter_grid_search_status.txt" ]; then
        rm "grid_search_files/parameter_grid_search_status.txt"
        echo "Deleted grid_search_files/parameter_grid_search_status.txt"
    else
        echo "grid_search_files/parameter_grid_search_status.txt does not exist."
    fi
}

# Prompt the user
prompt_user

# Check the user's response
if [[ "$response" =~ ^[Yy]$ ]]; then
    cd ..
    delete_directories_and_files
elif [[ "$response" =~ ^[Nn]$ ]]; then
    echo "No directories or files were deleted."
else
    echo "Invalid input. Please enter Y or N."
    prompt_user
fi
