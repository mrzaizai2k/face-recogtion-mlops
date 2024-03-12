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
from Utils.config_parser import read_config
from redis_database import *
from Utils.utils import *
# from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import pickle
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
                redis_database: RedisDatabase = RedisDatabase()):
        

        self.redis_database = redis_database
        self.redis_database.flush_database() # optional
        self.redis_database.flush_database_at_schedule() 

        self.green_color = (0, 255, 0)
        self.red_color = (0, 0, 255)
        self.active = True
        self.results = []

        #Load face liveness model
        self.face_liveness_config_path =face_liveness_config_path
        self.face_liveness_config = read_config(path=self.face_liveness_config_path)
        self.liveness_model = load_model(self.face_liveness_config.get('liveness_model_path'))
        self.liveness_label_encoder = pickle.loads(open(self.face_liveness_config.get('liveness_label_encoder_path'), "rb").read())

        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        #Load Face Recognition model
        self.recognization_config_path =recognization_config_path
        self.recognition_config = read_config(path=self.recognization_config_path)

        compre_face: CompreFace = CompreFace(host, port, self.recognition_config)
        self.recognition: RecognitionService = compre_face.init_face_recognition(api_key)
        self.FPS = 1/26

        # Start frame retrieval thread
        self.thread = Thread(target=self.show_frame, args=())
        self.thread.daemon = True
        self.thread.start()

    def check_and_update_empNo(self, empNo):
        '''Update employee to Redis cache to not spam the Noti API'''
        if self.redis_database.is_empno_exists(emp_no=empNo): 
            return False, self.green_color
        else:
            self.redis_database.update_empno(emp_no=empNo)
            return is_emp(empNo), self.red_color
        
    
        
    def draw_face_box(self, result, color:tuple = (0, 0, 255)):
        '''Draw face with information on screen
        color: Tuple - default is red
        '''
        box = result.get('box')
        age = result.get('age')
        gender = result.get('gender')
        mask = result.get('mask')
        subjects = result.get('subjects')

        
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

            if self.results:
                results = self.results
                for result in results:
                    box = result.get('box')
                    face = self.frame[box['y_min']:box['y_max'], box['x_min']:box['x_max']]
                    # cv2.imshow('face', face)
                    
                    liveness_preds, liveness_label = liveness_predict(face=face, 
                                                                    liveness_model=self.liveness_model,
                                                                    liveness_label_encoder=self.liveness_label_encoder,
                                                                    )
                    
                    print (f"liveness_label: {liveness_label} - liveness_preds: {liveness_preds}")
                    if (liveness_label == 'real') and (liveness_preds[1] >= self.face_liveness_config.get('acc_threshold')):
                        subjects = result.get('subjects')
                        # print ('liveness_label', liveness_label)
                        empNo = subjects[0]['subject'].split("_")[0]
                        # is_empNo_in_db, color = self.check_and_update_empNo(empNo=empNo)
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
        try:
            print("FPS: ", 1.0 / (time.time() - start_time))
        except:
            pass