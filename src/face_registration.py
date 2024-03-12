import sys
sys.path.append("")
from compreface import CompreFace
from compreface.service import RecognitionService
from compreface.collections import FaceCollection
from compreface.collections.face_collections import Subjects
import time
import os
import time
import json
import uuid
import requests
import base64
import numpy as np

from PIL import Image
import base64
from src.test_api import get_base64

from dotenv import load_dotenv
load_dotenv()
from Utils.utils import timeit, check_path, is_emp


DOMAIN: str = 'http://localhost'
PORT: str = '8000'
API_KEY: str = os.getenv('API_KEY')

compre_face: CompreFace = CompreFace(DOMAIN, PORT)

recognition: RecognitionService = compre_face.init_face_recognition(API_KEY)

face_collection: FaceCollection = recognition.get_face_collection()

subjects: Subjects = recognition.get_subjects()
# subject: str = 'Toan'

@timeit
def write_face_CNTT(folder_path = 'images/'):

    # Loop through the images in the folder

    for filename in os.listdir(folder_path):
        # if filename.endswith(('.jpg', '.jpeg', '.png')):  # Consider only image files
        image_path = os.path.join(folder_path, filename)
        subject = os.path.splitext(filename)[0]  # Use the filename as the subject (remove extension)
        subjects.delete(subject)
        # Add the face to the collection
        face_collection.add(image_path=image_path, subject=subject)

    print(subjects.list())
    print ('Done')

@timeit
def write_face_lfw(parent_folder_path = 'lfw-deepfunneled/lfw-deepfunneled'):
    for subfolder_name in os.listdir(parent_folder_path):
        subfolder_path = os.path.join(parent_folder_path, subfolder_name)

        # Step 2: Read the first image path in each folder
        image_files = [filename for filename in os.listdir(subfolder_path) 
                    if filename.endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            # Skip the folder if there are no image files
            continue

        first_image_path = os.path.join(subfolder_path, image_files[0])

        # Step 3: Use the subfolder name as the subject
        subject = subfolder_name

        # Step 4: Delete existing subject and add the face to the collection
        subjects.delete(subject)
        face_collection.add(image_path=first_image_path, subject=subject)

    print(subjects.list())
    print('Done')

    

@timeit
def recognize_face(image_path):
    base64_img = get_base64()
    image_data = base64.b64decode(base64_img)
    data = recognition.recognize(image_data)
    
    # data = recognition.recognize(image_path=image_path)
    print(f"Recognition Result: {data}")
    results = data['result']
    # Extract the subject
    subject = data['result'][0]['subjects'][0]['subject']

    # Print the extracted subject
    print('subject:', subject)
    print('results:', results)
    return results
    


if __name__ == "__main__":
    # write_face_CNTT()
    # write_face_lfw()
    recognize_face('test/2.PNG')
    print(is_emp(empNo='12062'))
