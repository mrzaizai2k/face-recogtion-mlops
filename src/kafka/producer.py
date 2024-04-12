import json
import re
import sys
sys.path.append("")
import time
from multiprocessing import Process

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream
from kafka import KafkaProducer
from src.kafka.utils import *


class StreamVideo(Process):

    def __init__(self, video_path,
                 topic,
                 use_cv2=False,
                 pub_obj_key="original", #should always be "original" in our case
                 group=None,
                 target=None,
                 camera_name=None,
                 verbose=False,):
        """Video Streaming Producer Process Class. Publishes frames from a video source to a topic.

        :param video_path: video path or url
        :param topic: kafka topic to publish stamped encoded frames.
        :param topic_partitions: number of partitions this topic has, for distributing messages among partitions
        :param use_cv2: send every frame, using cv2 library, else will use imutils to speedup training. use_cv2 = True is for camera, else is for videos
        :param pub_obj_key: associate tag with every frame encoded, can be used later to separate raw frames
        :param group: group should always be None; it exists solely for compatibility with threading.
        :param target: Process Target
        :param name: Process name
        :param verbose: print logs on stdout
        :param rr_distribute: use round robin partitioner, should be set same as consumers.

        """

        super().__init__(group=group, target=target, name=camera_name)

        # URL for streaming video
        self.video_path = video_path
        # TOPIC TO PUBLISH
        self.frame_topic = topic

        self.use_cv2 = use_cv2
        self.object_key = pub_obj_key
        self.verbose = verbose

    def run(self):
        """Publish video frames as json objects, timestamped, marked with camera number.

        Source:
            self.video_path: URL for streaming video
            self.kwargs["use_cv2"]: use raw cv2 streaming, set to false to use smart fast streaming --> not every frame is sent.
        Publishes:
            A dict {"frame": string(base64encodedarray), "dtype": obj.dtype.str, "shape": obj.shape,
                    "timestamp": time.time(), "camera": camera, "frame_num": frame_num}
        """


        # Producer object, set desired partitioner
        frame_producer = KafkaProducer(bootstrap_servers=["localhost:9092"],
                                       key_serializer=lambda key: str(key).encode(),
                                       value_serializer=lambda value: json.dumps(value).encode(),
                                       )

        print(f"[CAM {self.name}] URL: {self.video_path}, SET PARTITIONS FOR FRAME TOPIC: {self.frame_topic}")
        # Use either option
        video = cv2.VideoCapture(self.video_path) if self.use_cv2 else VideoStream(self.video_path).start()

        # Track frame number
        frame_num = 0
        start_time = time.time()
        print(f"[CAM {self.name}] START TIME {start_time}: ")

        # Read URL, Transform, Publish
        while True:

            # using raw cv2, frame by frame
            if self.use_cv2:
                success, image = video.read()
                # check if the file has read
                if not success:
                    if self.verbose:
                        print(f"[CAM {self.name}] URL: {self.video_path}, END FRAME: {frame_num}")
                    break

            # using smart, only unique frames, skips frames, faster fps
            else:
                image = video.read()
                # check if the file has read
                if image is None:
                    if self.verbose:
                        print(f"[CAM {self.name}] URL: {self.video_path}, END FRAME: {frame_num}")
                    break
            

            # Attach metadata to frame, transform into JSON
            message = self.transform(frame=image,
                                     frame_num=frame_num,
                                     object_key=self.object_key,
                                     camera_name=self.name,
                                     verbose=self.verbose)


            # Publish to specific partition
            frame_producer.send(self.frame_topic, key=f"{self.name}_{frame_num}", value=message)

            # if frame_num % 1000 == 0:
            frame_producer.flush()

            frame_num += 1

        # clear the capture
        if self.use_cv2:
            video.release()
        else:
            video.stop()

        if self.verbose:
            print(f"[CAM {self.name}] FINISHED. STREAM TIME {time.time() - start_time}: ")

        return True if frame_num > 0 else False

    @staticmethod
    def transform(frame, frame_num, object_key="original", camera_name="192.168.1.2", verbose=False):
        """Serialize frame, create json message with serialized frame, camera number and timestamp.

        :param frame: numpy.ndarray, raw frame
        :param frame_num: frame number in the particular video/camera
        :param object_key: identifier for these objects
        :param camera: Camera Number the frame is from
        :param verbose: print out logs
        :return: A dict {"frame": string(base64encodedarray), "dtype": obj.dtype.str, "shape": obj.shape,
                    "timestamp": time.time(), "camera": camera, "frame_num": frame_num}
        """
        frame = imutils.resize(frame, width=480)

        if verbose:
            # print raw frame size
            print("\nRAW ARRAY SIZE: ", sys.getsizeof(frame))

        # serialize frame
        frame_dict = np_to_json(frame.astype(np.uint8), prefix_name=object_key)
        # Metadata for frame
        message = {"timestamp": time.time(), "camera": camera_name, "frame_num": frame_num}
        # add frame and metadata related to frame
        message.update(frame_dict)

        if verbose:
            # print message size
            print("\nMESSAGE SIZE: ", sys.getsizeof(message))

        return message
    
if __name__=="__main__":
    # test/kafka_videos/video-1.mp4
    producer = StreamVideo(video_path="test/kafka_videos/video-1.mp4", topic="message", camera_name="192.168.2.3")
    producer.run()
