import json
import sys
sys.path.append("")
import socket
import time
from contextlib import contextmanager
from multiprocessing import Process

import cv2
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from kafka.structs import OffsetAndMetadata, TopicPartition


from src.Utils.utils import *
from src.kafka.utils import *


def face_recognition_bytes(np_array, root_url="http://10.18.25.16:8080"):
    url = f"{root_url}/v1/recognize"
    img = np_array.copy()
    _, img_buffer_arr = cv2.imencode(".jpg", img)
    img_bytes = img_buffer_arr.tobytes()
    
    headers = {'Content-Type': 'image/jpeg'}
    res = requests.post(url, data=img_bytes, headers=headers)
    print(f'result: {res.json()}')


class PredictFrames(Process):

    def __init__(self,
                 processed_frame_topic,
                 verbose=False,
                 group=None,
                 target=None,
                 name=None):
        """
        FACE MATCHING TO QUERY FACES --> Consuming frame objects to produce predictions.

        :param processed_frame_topic: kafka topic to consume from stamped encoded frames with face detection and encodings.
        :param query_faces_topic: kafka topic which broadcasts query face names and encodings.
        :param scale: (0, 1] scale used during pre processing step.
        :param verbose: print log
        :param rr_distribute: use round robin partitioner and assignor, should be set same as respective producers or consumers.
        :param group: group should always be None; it exists solely for compatibility with threading.
        :param target: Process Target
        :param name: Process name
        """
        super().__init__(group=group, target=target, name=name)

        self.iam = "{}-{}".format(socket.gethostname(), self.name)
        self.frame_topic = processed_frame_topic
        self.verbose = verbose
        print("[INFO] I am ", self.iam)

    def run(self):
        """Consume pre processed frames, match query face with faces detected in pre processing step
        (published to processed frame topic) publish results, box added to frame data if in params,
        ORIGINAL_PREFIX == PREDICTED_PREFIX"""

        frame_consumer = KafkaConsumer(group_id="face_recognition",
                                       bootstrap_servers=["localhost:9092"],
                                       key_deserializer=lambda key: key.decode(),
                                       value_deserializer=lambda value: json.loads(value.decode()),
                                       auto_offset_reset="earliest",
                                       )

        frame_consumer.subscribe([self.frame_topic])


        try:

            while True:

                if self.verbose:
                    print("[PredictFrames {}] WAITING FOR NEXT FRAMES..".format(socket.gethostname()))

                raw_frame_messages = frame_consumer.poll(timeout_ms=10, max_records=10)

                for topic_partition, msgs in raw_frame_messages.items():
                    # Get the predicted Object, JSON with frame and meta info about the frame
                    for msg in msgs:
                        # print('msg', msg)
                        tp = TopicPartition(msg.topic, msg.partition)
                        offsets = {tp: OffsetAndMetadata(msg.offset, None)}
                        frame_consumer.commit(offsets=offsets)

                        print(f'partition: {msg.partition}, offset: {msg.offset}, timestamp: {msg.timestamp}')
                        base64_img = msg.value["original_frame"]
                        img = np_from_json(msg.value, prefix_name="original") 
                        # img = cv2.cvtColor(img.astype(np.uint8))
                        # face_recognition_bytes(np_array=img)
                        # img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                        cv2.imshow('img', img)
                
                cv2.waitKey(1)

        except KeyboardInterrupt as e:
            print("Closing Stream")
            frame_consumer.close()
            if str(self.name) == "1":
                pass

        finally:
            print("Closing Stream")
            frame_consumer.close()


@contextmanager
def timer(name):
    """Util function: Logs the time."""
    t0 = time.time()
    yield
    print("[{}] done in {:.3f} s".format(name, time.time() - t0))

if __name__=="__main__":
    producer = PredictFrames( processed_frame_topic="message", name="video-1")
    producer.run()
