import json 
from kafka import KafkaConsumer
import cv2
import numpy as np 

if __name__ == '__main__':
    # Kafka Consumer 
    consumer = KafkaConsumer(
        'message',
        bootstrap_servers='localhost:9092',
        # auto_offset_reset='earliest',
    )
    for message in consumer:
        try:
            print('message.header', message.headers)
            my_bytes_value = message.value
            
            # Decode the byte array to a numpy array representing the image
            nparr = np.frombuffer(my_bytes_value, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Display the image
            cv2.imshow('Received Image', img)
            
            # Wait for a key event for 1 millisecond
            # If any key is pressed, close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            pass



# from flask import Flask, render_template, request
# from flask_socketio import SocketIO
# from random import random
# from threading import Lock
# from datetime import datetime
# import threading

# from flask import Flask, render_template
# from flask_socketio import SocketIO
# # from utils import insert_data_unique, create_collections_unique
# import base64
# import requests
# from io import BytesIO
# import io
# import numpy as np
# import cv2
# import json
# import time

# thread = None
# thread_lock = Lock()

# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'ajith!'
# socketio = SocketIO(app, cors_allowed_origins='*')

# def background_thread():
#     client = MongoClient(host='mongodb_connect',
#                          port=27017, 
#                          username='root', 
#                          password='pass',
#                         authSource="admin")
#     db = client['stream-database']
#     print("connected with db")
#     video_names = ["video0", "video1", "video2"]
#     videos_map = create_collections_unique(db, video_names)
    
#     print("Generating random sensor values")
#     consumer = Consumer(consumer_config)
#     consumer.subscribe(["videostreaming"])
#     while True:
#         msg = consumer.poll(timeout = 0.1)
#         if msg == None:
#             # print(" data not received successfully")
#             socketio.emit('update_ui', {'data': "data didnt successfully"})
#         elif msg.error() == None:
#             # print("received")
#             # convert image bytes data to numpy array of dtype uint8
#             nparr = np.frombuffer(msg.value(), np.uint8)

#             # decode image
#             img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#             img = cv2.resize(img, (320, 320))
#             _, img_byte_arr = cv2.imencode('.jpg', img)
#             img_byte_arr = img_byte_arr.tobytes()
#             payload = {'data': img_byte_arr}


#             out = requests.post("http://localhost:8080/predictions/yolov8n", data=payload)
#             # files = {'data': img.tolist()}

#             # outputs = requests.post("http://localhost:8080/predictions/yolov8x", files=files)
#             # outputs = outputs.json()
#             outputs = out.text
#             outputs = json.loads(outputs)


#             print(outputs)
#             # Loop through each bounding box and draw on the image
#             for output in outputs:
#                 x, y, w, h = map(int, output["box"])
#                 class_name = output["class_name"]
#                 confidence = output["confidence"]

#                 # Draw bounding box on the image
#                 cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

#                 # Display class name and confidence
#                 text = f"{class_name}: {confidence:.2f}"
#                 cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#             _, img_base64 = cv2.imencode('.jpg', img)
#             img_base64_str = base64.b64encode(img_base64).decode('utf-8')

#             # get metadata
#             frame_no = msg.timestamp()[1]
#             video_name = msg.headers()[0][1].decode("utf-8")
#             dataframes = {}
#             # print("frame_no-",frame_no)
#             # print("video_name", video_name)
#             dataframes["frame"] = frame_no
#             dataframes["videoname"] = video_name
#             dataframes["predictions"] = outputs
            
#             if (video_name=="video0"):
#                 socketio.emit('passimage0', {'data':[video_name, img_base64_str,  frame_no]}) #mit
#             elif (video_name=="video1"):
#                 socketio.emit('passimage1', {'data':[video_name, img_base64_str,  frame_no]}) #mit

#             # commit synchronously
#             videos_map[video_name] = [dataframes]
#             insert_data_unique(db, videos_map)
#             consumer.commit(asynchronous=False)
#         elif msg.error().code() == KafkaError._PARTITION_EOF:
#                             print('End of partition reached {0}/{1}'
#                                 .format(msg.topic(), msg.partition()))
#         else:
#             print('Error occured: {0}'.format(msg.error().str()))

# """
# Serve root index file
# """
# @app.route('/')
# def index():
#     return render_template('index.html')



# """
# Decorator for connect
# """
# @socketio.on('connect')
# def connect():
#     global thread
#     print('Client connected')

#     global thread
#     with thread_lock:
#         if thread is None:
#             thread = socketio.start_background_task(background_thread)
# """
# Decorator for disconnect
# """
# @socketio.on('disconnect')
# def disconnect():
#     print('Client disconnected',  request.sid)

# if __name__ == '__main__':
    
#     socketio.run(app,host='0.0.0.0',allow_unsafe_werkzeug=True)


# # # 