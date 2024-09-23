"""
Script to hold general helper functions
"""


import sys
import traceback
from datetime import datetime, timezone, timedelta
import io
import re


def get_time_stamp():
    """
    YYYY_MM_DD_HH_MM_SS_MS time stamp that can be added to filenames to sort them
    in order and make them unique.
    
    Parameters:
        None
    Returns:
        formatted_time (str): formatted time string YYYY_MM_DD_HH_MM_SS_MS
    """
    # Get current time in UTC
    current_time_utc = datetime.now(timezone.utc)
    
    # Convert UTC time to Central Time Zone (UTC-6:00 for standard time, UTC-5:00 for daylight saving time)
    central_time_offset = timedelta(hours=-5)  # Adjust for daylight saving time if necessary
    current_time_central = current_time_utc + central_time_offset
    
    # Format the time as specified
    formatted_time = current_time_central.strftime("%Y_%m_%d__%H_%M_%S_%f")
    # only keep the first 2 digits of milliseconds
    formatted_time = formatted_time[:-4]
    return formatted_time


def convert_seconds_to_readable_time_str(seconds):
    """
    converts seconds to hours, minutes, and seconds.  Helpful for printing out training times.
    
    returns a string in the format: "HH:MM:SS" prepended with an explanation "HH:MM:SS"
    example: "HH:MM:SS::00:00:00"
    
    Parameters:
        seconds (int): time in seconds to convert
    Returns:
        formatted_time (str): formatted time string
    """
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"HH:MM:SS::{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def convert_readable_time_str_to_seconds(time_str):
    """
    Convert a readable time string in the format 'HH:MM:SS' to seconds.
    
    Parameters:
    time_str (str): A string representing the time in the format 'HH:MM:SS'.
    
    Returns:
    int: The total number of seconds.
    """
    # Split the time string and extract the hours, minutes, and seconds
    time_parts = time_str.split(':')[4:]
    
    # Convert hours, minutes, and seconds to integers
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    seconds = int(time_parts[2])
    
    # Calculate the total seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds


class Terminal_Logger:
    """
    Custom class to redirect the standard output and standard error to a log file while still printing to the terminal. Useful for running
    scripts on remote servers that might close the terminal session and lose the output or for viewing the output of older scripts.
    """
    def __init__(self, log_path):
        """
        log_path: PurePath object representing the full path of the log file
        """
        self.terminal = sys.stdout
        self.log = io.open(log_path, 'w', encoding='utf-8')
        self._setup_logging()
        print(f"Logging terminal output to: {log_path}")
        return
    
    def _setup_logging(self):
        sys.stdout = self
        sys.stderr = self
        sys.excepthook = self.custom_exception_hook
        return
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure messages are written out immediately
        return
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        return
    
    def custom_exception_hook(self, exctype, value, tb):
        """
        custom exception hook to print the error traceback to stderr, which is now redirected to Logger. When the script errors out, the
        logs will contain the error traceback.
        """
        # Print the error traceback to stderr, which is now redirected to Logger
        print(f"\n\nERROR OCCURRED\n{'-'*40}", file=sys.stderr)
        traceback.print_exception(exctype, value, tb)
        return
    
    def reconnect_to_log_file(self):
        """
        Re-establish logging to the log file. This is useful if sys.stdout was redirected temporarily.
        """
        self._setup_logging()
        return
