import sys
sys.path.append("")
import numpy as np
import json
import requests
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
import base64
from io import BytesIO
from pprint import pprint



def convert_image_to_base64(image_path):
    try:
        with open(image_path, 'rb') as img_file:
            img = Image.open(img_file)
            # Convert the image to RGB (in case it's a different mode)
            img = img.convert('RGB')
            buffered = BytesIO()
            img.save(buffered, format="JPEG")  # Change format to match your image type
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return img_str
    except FileNotFoundError:
        print("The specified file was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def get_base64(img_path):
    base64_img =  convert_image_to_base64(img_path)
    return base64_img

def test_face_recognition_api(img_path, root_url):
    url = f"{root_url}/recognize"
    base64_img = get_base64(img_path)
    res = requests.post(url, json ={"img":base64_img})
    # print(f'Status Code: {res.status_code}')
    print(f'result: {res.json()}')

def test_add_face_api(img_path, root_url):

    url = f"{root_url}//add_face"

    base64_img = get_base64(img_path)

    test_data = {
        "employee_No": "12062",
        "img": base64_img  # Replace with your actual base64 encoded image
    }

    # API endpoint
    
    # Send POST request
    response = requests.post(url, json=test_data)

    # Check response status code
    if response.status_code == 201:
        print(f'response: {response.json()}')
    else:
        print(f"Error response: {response.json()}")
        

if __name__ == "__main__":
    img_path = 'test/3.jpg'
    root_url = "http://10.18.25.16:8080"

    test_face_recognition_api(img_path, root_url)
    test_add_face_api(img_path, root_url)
