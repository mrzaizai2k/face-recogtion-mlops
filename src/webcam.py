import sys
sys.path.append("")
import os
from dotenv import load_dotenv
load_dotenv()

import cv2
import argparse
import time
from threading import Thread

from compreface import CompreFace
from compreface.service import RecognitionService
from Utils.utils import *
# from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import pickle
import tensorflow as tf
from liveness import *


def parseArguments():
    parser = argparse.ArgumentParser()
    API_KEY: str = os.getenv('API_KEY')
    parser.add_argument("--api-key", help="CompreFace recognition service API key", type=str, default=API_KEY)
    parser.add_argument("--host", help="CompreFace host", type=str, default='http://localhost')
    parser.add_argument("--port", help="CompreFace port", type=str, default='8000')
    args = parser.parse_args()

    return args


class ThreadedCamera:
    def __init__(self, api_key, host, port, 
                recognization_config_path = 'config/face_recognization.yaml', 
                face_liveness_config_path = 'config/face_liveness.yaml', 
                ):
        

        self.green_color = (0, 255, 0)
        self.red_color = (0, 0, 255)
        self.active = True
        self.results = []

        print("[INFO] loading liveness detector...")

        TF_MODEL_FILE_PATH = 'models/liveness/binary_model.tflite' # The default path to the saved TensorFlow Lite model

        self.interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
        print(self.interpreter.get_signature_list())

        self.classify_lite = self.interpreter.get_signature_runner('serving_default')
        print("classify_lite", self.classify_lite)
        self.class_names = ['fake', 'real']

        self.capture = cv2.VideoCapture("test_data/video1.mp4")
        # self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        #Load Face Recognition model
        self.recognization_config_path =recognization_config_path
        self.recognition_config = read_config(path=self.recognization_config_path)

        compre_face: CompreFace = CompreFace(host, port, self.recognition_config)
        self.recognition: RecognitionService = compre_face.init_face_recognition(api_key)
        self.FPS = 1/26
        self.faces = []
        # Start frame retrieval thread
        self.thread = Thread(target=self.show_frame, args=())
        self.thread.daemon = True
        self.thread.start()
    
        
    def draw_face_box(self, result, color:tuple = (0, 0, 255)):
        '''Draw face with information on screen
        color: Tuple - default is red
        '''
        # print("result", result)
        box = result.get('box')
        age = result.get('age')
        gender = result.get('gender')
        mask = result.get('mask')
        pose = result.get('pose')
        landmarks = result.get('landmarks')
        subjects = result.get('subjects')

        # print("mask", mask)
        # print("pose", pose)
        # print("landmarks", landmarks)

        if box:
            cv2.rectangle(img=self.frame, pt1=(box['x_min'], box['y_min']),
                                        pt2=(box['x_max'], box['y_max']), color = color, thickness=1)
            if age:
                age = f"Age: {age['low']} - {age['high']}"
                cv2.putText(self.frame, age, (box['x_max'], box['y_min'] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            if gender:
                gender = f"Gender: {gender['value']}"
                cv2.putText(self.frame, gender, (box['x_max'], box['y_min'] + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            if mask:
                mask = f"Mask: {mask['value']}"
                cv2.putText(self.frame, mask, (box['x_max'], box['y_min'] + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

            if subjects:
                subjects = sorted(subjects, key=lambda k: k['similarity'], reverse=True)
                subject = f"Subject: {subjects[0]['subject']}"
                similarity = f"Similarity: {subjects[0]['similarity']}"
                cv2.putText(self.frame, subject, (box['x_max'], box['y_min'] + 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                cv2.putText(self.frame, similarity, (box['x_max'], box['y_min'] + 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            # if liveness_label:
            #     cv2.putText(self.frame, liveness_label, (box['x_max'], box['y_min'] + 15),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

            else:
                subject = f"No known faces"
                cv2.putText(self.frame, subject, (box['x_max'], box['y_min'] + 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                

    def show_frame(self):
        print("Started")
        while self.capture.isOpened():
            (status, frame_raw) = self.capture.read()
            self.frame = cv2.flip(frame_raw, 1)
            self.frame_height, self.frame_width, _ = self.frame.shape
            pose_threshold = 10 

            if self.results:
                print("self.results", self.results)
                results = self.results
                for result in results:
                    print('result', result)
                        
                    box = result.get('box')
                    mask = result.get('mask')
                    pose = result.get('pose')
                    landmarks = result.get('landmarks')

                    if mask['value'] != "without_mask":
                        print("mask", mask)
                        continue  # Skip to the next iteration of the loop

                            # Calculate face height
                    face_height = box['y_max'] - box['y_min']

                    
                    # Check if face height is less than 1/4 of the video height. The face is too small
                    if face_height < self.frame_height / 5:
                        print("face_height", face_height)
                        continue

                    if abs(pose['pitch']) > 13 or abs(pose['roll']) > 13 or abs(pose['yaw']) > 30:
                        continue  # Skip to the next iteration of the loop


                    face = self.frame[box['y_min']:box['y_max'], box['x_min']:box['x_max']]
                    face = cv2.resize(face, (32, 32))
                    # Convert the resized face image to a format compatible with TensorFlow
                    face = tf.keras.preprocessing.image.img_to_array(face)
                    face = tf.expand_dims(face, 0)  # Create a batch

                    # cv2.imshow('face', face)

                    predictions_lite = self.classify_lite(input_2=face)['fc2']
                    score_lite = predictions_lite[0]
                    
                    if score_lite[0] > 0.5:
                        self.faces.append(result)
                        self.draw_face_box(result, color = self.green_color)


            cv2.imshow('CompreFace demo', self.frame)
            time.sleep(self.FPS)

            if cv2.waitKey(1) & 0xFF == 27:
                self.capture.release()
                cv2.destroyAllWindows()
                self.active=False

    def is_active(self):
        return self.active

    def update_frame(self):
        if not hasattr(self, 'frame'):
            return

        _, im_buf_arr = cv2.imencode(".jpg", self.frame)
        byte_im = im_buf_arr.tobytes()
        data = self.recognition.recognize(byte_im)
        self.results = data.get('result')


if __name__ == '__main__':
    args = parseArguments()
    threaded_camera = ThreadedCamera(args.api_key, args.host, args.port)
    while threaded_camera.is_active():
        start_time = time.time()
        threaded_camera.update_frame()
        # try:
        #     print("FPS: ", 1.0 / (time.time() - start_time))
        # except:
        #     pass