import os
from compreface import CompreFace
from compreface.service import RecognitionService
from compreface.collections import FaceCollection
from compreface.collections.face_collections import Subjects
from dotenv import load_dotenv
load_dotenv()


DOMAIN: str = 'http://localhost'
PORT: str = '8000'

API_KEY = os.getenv('API_KEY')


compre_face: CompreFace = CompreFace(DOMAIN, PORT)

recognition: RecognitionService = compre_face.init_face_recognition(API_KEY)

face_collection: FaceCollection = recognition.get_face_collection()

subjects: Subjects = recognition.get_subjects()

image_path: str = 'data/face_images/Mai_Chi_Bao.jpg'
subject: str = 'Mai_Chi_Bao'

face_collection.add(image_path=image_path, subject=subject)