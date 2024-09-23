@REM script to reset the project's saved data directories to a clean state runnable on Windows

@echo off
setlocal

:prompt
set /p response=reset all save directories in this project? [Y/n]: 
if /i "%response%"=="" set response=Y
if /i "%response%"=="Y" goto proceed
if /i "%response%"=="N" goto end
echo Invalid input. Please enter Y or N.
goto prompt

:proceed
@REM navigate to the directory where this script is located
cd %~dp0
@REM navigate up one directory to the project root
cd ..

if exist mlflow_data_storage\ (
    del /q mlflow_data_storage\*
    echo Deleted all files in mlflow_data_storage
) else (
    echo mlflow_data_storage does not exist.
)

if exist model_and_metrics_saves\ (
    del /q model_and_metrics_saves\*
    echo Deleted all files in model_and_metrics_saves
) else (
    echo model_and_metrics_saves does not exist.
)

if exist grid_search_files\parameter_grid.csv (
    del /q grid_search_files\parameter_grid.csv
    echo Deleted grid_search_files\parameter_grid.csv
) else (
    echo grid_search_files\parameter_grid.csv does not exist.
)

if exist grid_search_files\parameter_grid_search_status.txt (
    del /q grid_search_files\parameter_grid_search_status.txt
    echo Deleted grid_search_files\parameter_grid_search_status.txt
) else (
    echo grid_search_files\parameter_grid_search_status.txt does not exist.
)

echo All data and directories inside "mlflow_data_storage" and "model_and_metrics_saves" have been deleted.
echo Specific files in "grid_search_files" have been checked and deleted if they existed.

:end
endlocal
