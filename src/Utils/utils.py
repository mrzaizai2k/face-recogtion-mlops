from functools import wraps
import time
from datetime import datetime
import os
import psutil
import json
import uuid
import yaml
import requests
from dotenv import load_dotenv
load_dotenv()


def convert_ms_to_hms(ms):
    seconds = ms / 1000
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    seconds = round(seconds, 2)
    
    return f"{int(hours)}:{int(minutes):02d}:{seconds:05.2f}"

def measure_memory_usage(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        mem_start = process.memory_info()[0]
        rt = func(*args, **kwargs)
        mem_end = process.memory_info()[0]
        diff_MB = (mem_end - mem_start) / (1024 * 1024)  # Convert bytes to megabytes
        print('Memory usage of %s: %.2f MB' % (func.__name__, diff_MB))
        return rt
    return wrapper


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        time_delta = convert_ms_to_hms(total_time*1000)

        print(f'{func.__name__.title()} Took {time_delta}')
        return result
    return timeit_wrapper

def is_file(path:str):
    return '.' in path

def check_path(path):
    # Extract the last element from the path
    last_element = os.path.basename(path)
    if is_file(last_element):
        # If it's a file, get the directory part of the path
        folder_path = os.path.dirname(path)

        # Check if the directory exists, create it if not
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Create new folder path: {folder_path}")
        return path
    else:
        # If it's not a file, it's a directory path
        # Check if the directory exists, create it if not
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Create new path: {path}")
        return path

def read_config(path = 'config/config.yaml'):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def rename_model(model_dir:str='models/liveness/weights', prefix:str="facenet"):
    """
    Adds a prefix with date (YYYY_mm_dd) and sequential ID to filenames in a directory.

    Args:
        model_dir (str): Path to the directory containing models.
        prefix (str, optional): Prefix to add before the date and ID (default: "vit_teacher").

    Returns:
        str: The final model filename with prefix and ID.
    """

    # Get the model directory path (from config if available, otherwise use the provided path)

    # Ensure the directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Get today's date in YYYY_mm_dd format
    today_str = datetime.now().strftime("%Y_%m_%d")

    # Start with ID 1
    id = 1

    # Construct the base filename with prefix and date
    base_filename = f"{prefix}_{today_str}"

    while True:
        # Build the full filename with ID
        filename = f"{base_filename}_{id}.pth"

        # Check if the file already exists
        if not os.path.exists(os.path.join(model_dir, filename)):
            return os.path.join(model_dir, filename)

        # Increment ID for next attempt
        id += 1
