from functools import wraps
import time
from datetime import datetime
import os
import psutil
import json
import uuid
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

from sklearn.datasets import fetch_lfw_people
import os
from os.path import join, exists
import matplotlib.pyplot as plt

def save_limited_lfw_faces(lfw_path="lfw_faces", num_people=50):
    """Saves one face for each of the specified number of unique people.

    Args:
        lfw_path (str, optional): Path to save the LFW faces folder. Defaults to "lfw_faces".
        num_people (int, optional): The number of unique people's faces to download and save. Defaults to 500.
    """
    # Fetch the LFW dataset
    lfw_people = fetch_lfw_people(min_faces_per_person=50, resize=0.5)
    images, targets, target_names = lfw_people.images, lfw_people.target, lfw_people.target_names

    if not exists(lfw_path):
        os.mkdir(lfw_path)

    saved_people = set()  # Track the individuals who have already had a face saved
    num_saved = 0  # Counter for the number of saved faces

    for i, (image, target) in enumerate(zip(images, targets)):
        if num_saved >= num_people:
            break  # Stop once we've saved the desired number of people's faces

        name = target_names[target].replace(" ", "_")  # Create a valid folder/file name

        # Save an image only if this person hasn't been saved yet
        if name not in saved_people:
            person_folder = join(lfw_path, name)

            if not exists(person_folder):
                os.mkdir(person_folder)

            # Save the image with the person's name as the filename
            image_path = join(person_folder, f"{name}.png")

            # Save the image using matplotlib's plt.imsave
            plt.imsave(image_path, image, cmap='gray')

            saved_people.add(name)  # Mark this person as saved
            num_saved += 1  # Increment the saved counter

            if num_saved % 100 == 0 or num_saved == num_people:
                print(f"Saved {num_saved} unique faces...")

    print(f"Total of {num_saved} unique faces saved to {lfw_path}.")


