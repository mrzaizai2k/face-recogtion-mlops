import sys
sys.path.append("")
import time 
import json 
import random 
from datetime import datetime
from kafka import KafkaProducer

# Messages will be serialized as JSON 
# def serializer(message):
#     return json.dumps(message).encode('utf-8')

# # Kafka Producer
# producer = KafkaProducer(
#     bootstrap_servers=['localhost:9092'],
#     value_serializer=serializer
# )

# if __name__ == '__main__':
#     # Infinite loop - runs until you kill the program
#     while True:
#         # Generate a message
#         dummy_message = generate_message()
        
#         # Send it to our 'messages' topic
#         print(f'Producing message @ {datetime.now()} | Message = {str(dummy_message)}')
#         producer.send('messages', dummy_message)
        
#         # Sleep for a random number of seconds
#         time_to_sleep = random.randint(1, 11)
#         time.sleep(time_to_sleep)


from glob import glob
import concurrent.futures
from src.kafka.utils import *
import os
import cv2
import time

class ProducerThread:
    def __init__(self):
        # self.producer = Producer(config)
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            # value_serializer=serializer
        )

    def publishFrame(self, video_path):
        video = cv2.VideoCapture(video_path)
        video_name = os.path.basename(video_path).split(".")[0]
        frame_no = 1
        while video.isOpened():
            _, frame = video.read()
            # pushing every 3rd frame
            if frame_no % 3 == 0:
                print(frame_no)
                frame_bytes = serializeImg(frame)
                self.producer.send(
                    topic="message", 
                    value=frame_bytes, 
                    # timestamp_ms=frame_no,
                    # headers={
                    #     "video_name": str.encode(video_name)
                    # }
                )
                # self.producer.poll(0)
            time.sleep(1)
            frame_no += 1
            
        video.release()
        return
        
    def start(self, vid_paths):
        # runs until the processes in all the threads are finished
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.publishFrame, vid_path) for vid_path in vid_paths]
            concurrent.futures.wait(futures)  # Wait for all threads to finish

        self.producer.flush() # push all the remaining messages in the queue
        print("Finished...")



if __name__ == "__main__":
    video_dir = "test_data/"
    video_paths = glob(video_dir + "*.mp4") # change extension here accordingly
    print(video_paths)
    producer_thread = ProducerThread()
    producer_thread.start(video_paths)
    