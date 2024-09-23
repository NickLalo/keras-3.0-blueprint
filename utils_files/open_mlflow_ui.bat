REM Activate the conda environment
call activate keras-env

REM logic to allow this script to be run from the main directory or this bat file's directory location and redirect to the correct location
REM if no utils_files directory (the directory where this file is located) exists in the current directory...
if not exist utils_files (
    REM check if the parent directory has a utils_files directory
    if exist ..\utils_files (
        REM navigate to the parent directory
        cd ..
    )
)

echo Delayed opening MLflow web app in Chrome in 7 seconds...
REM Run the commands in a new terminal window to open up the MLflow UI in Chrome after a short delay
start cmd /k "echo Delayed opening of MLflow in 7 seconds... & timeout /t 7 /nobreak & start chrome http://127.0.0.1:5000 & exit"

REM start up the mlflow local server with the storage location set to the mlflow_data_storage directory
echo Starting MLflow UI as a locally hosted web app...
mlflow ui --backend-store-uri mlflow_data_storage
