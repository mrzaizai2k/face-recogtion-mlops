import warnings # nopep8
warnings.filterwarnings("ignore") # nopep8
import sys # nopep8
sys.path.append("") # nopep8

import os
from flask import Flask, jsonify, request, make_response
import argparse

import base64
from io import BytesIO
from PIL import Image

from compreface import CompreFace
from compreface.service import RecognitionService
from compreface.collections import FaceCollection
from compreface.collections.face_collections import Subjects

from Utils.config_parser import read_config

app = Flask(__name__)

DOMAIN: str = 'http://localhost'
PORT: str = '8000'
API_KEY: str = os.getenv('API_KEY')
recognition_config = read_config(path='config/face_recognization.yaml')
compre_face: CompreFace = CompreFace(DOMAIN, PORT, recognition_config)

recognition: RecognitionService = compre_face.init_face_recognition(API_KEY)
face_collection: FaceCollection = recognition.get_face_collection()


@app.route('/recognize', methods=['POST'])
def recognize_face():
    """
    Route: /recognize (POST)

    Description: This API endpoint is used to recognize a face in an image.

    Request:

    Method: POST
    Content-Type: application/json
    Body:
        JSON
        {
            "img": "YOUR_BASE64_IMAGE"
        }

    Response:

    - Success (200): results: The staff ID of the recognized face.
    - Error (500): error: A string containing the error message.
    """
    try:
        # Get base64 image from the request
        base64_img = request.json['img']

        # # Convert base64 to bytes
        image_data = base64.b64decode(base64_img)
        data = recognition.recognize(image_data)
        print ('data',data)
        subject = data['result'][0]['subjects'][0]['subject']
        staff_cif = subject.split("_")[0]

        return jsonify({'results': staff_cif}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/add_face', methods=['POST'])
def add_face():
    """
    Route: /add_face (POST)

    Description: This API endpoint is used to add a new face to a collection.

    Request:

    Method: POST
    Content-Type: application/json
    Body:
    JSON
        {
        "employee_No": "12062",
        "img": "YOUR_BASE64_IMAGE"
        }

    employee_No: (Required) The employee number of the person whose face is being added.
    img: (Required) A base64 encoded string of the image containing the face to be added.
    Response:

    - Success (201):
    message: A message indicating that the face was added successfully, along with the subject name.
    - Error (400):
    error: A message indicating that required data is missing.
    - Error (500):
    error: A string containing the error message.
    """
    employee_No = request.json['employee_No']
    base64_image = request.json['img']

    if not all([employee_No, base64_image]):
        return jsonify({'error': 'Missing required data'}), 400

    try:
        # Decode base64 image and save as RGB
        img_data = base64.b64decode(base64_image)
        img = Image.open(BytesIO(img_data)).convert('RGB')

        # Create temporary folder if it doesn't exist
        tmp_folder = 'tmp'
        os.makedirs(tmp_folder, exist_ok=True)

        # Save image to temporary folder
        img_path = os.path.join(tmp_folder, f'{employee_No}.jpg')
        img.save(img_path)

        # Add face to collection
        subject = f'{employee_No}'
        face_collection.add(image_path=img_path, subject=subject)  # Assuming face_collection is available

        # Delete temporary folder
        os.remove(img_path)  # Remove individual image file
        os.rmdir(tmp_folder)

        return jsonify({'message': f'Face added successfully for {subject}'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=8080,
        help='Port of serving api')
    args = parser.parse_args()
    # serve(app, host='0.0.0.0', port=args.port)
    app.run(host='0.0.0.0', port=args.port)
