import cv2
import os
import numpy as np
from typing import Tuple
import insightface
from insightface.app import FaceAnalysis
import gdown
from pathlib import Path
from typing import List
from insightface.utils import face_align

class ModelDownloader:
    def __init__(self):
        self.model_dir = "models"
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

    def download_model(self, model_name: str, file_id: str) -> str:
        model_path = os.path.join(self.model_dir, model_name)
        if not os.path.exists(model_path):
            print(f"Downloading {model_name} model...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
            print(f"Model downloaded to {model_path}")
        else:
            print(f"Model {model_name} already exists at {model_path}")
        return model_path
    
class ImgScaler:
    def __init__(self, img_length_limit: int):
        self._img_length_limit = img_length_limit
        self._downscale_img_called = False
        self._downscale_coefficient = None

    def downscale_img(self, img, interpolation=cv2.INTER_AREA):
        assert not self._downscale_img_called
        self._downscale_img_called = True
        height, width = img.shape[:2]
        if width <= self._img_length_limit and height <= self._img_length_limit or not self._img_length_limit:
            return img

        self._downscale_coefficient = self._img_length_limit / (width if width >= height else height)
        new_width = round(width * self._downscale_coefficient)
        new_height = round(height * self._downscale_coefficient)
        return cv2.resize(img, dsize=(new_width, new_height), interpolation=interpolation)

    @property
    def downscale_coefficient(self) -> float:
        assert self._downscale_img_called
        if not self._downscale_coefficient:
            return 1

        return self._downscale_coefficient
    @property
    def upscale_coefficient(self) -> float:
        assert self._downscale_img_called
        if not self._downscale_coefficient:
            return 1

        return 1 / self._downscale_coefficient
    

class DetectionOnlyFaceAnalysis(FaceAnalysis):
        rec_model = None
        ga_model = None

        def __init__(self, file):
            self.det_model = face_detection.FaceDetector(file, 'net3')

import attr

class JSONEncodable:
    def to_json(self):
        if hasattr(self, 'dto'):
            return self.dto.to_json()
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
from typing import List, Tuple

import attr



# noinspection PyUnresolvedReferences
@attr.s(auto_attribs=True, frozen=True)
class BoundingBoxDTO(JSONEncodable):
    """
    >>> BoundingBoxDTO(x_min=10, x_max=0, y_min=100, y_max=200, probability=0.5)
    Traceback (most recent call last):
    ...
    ValueError: 'x_min' must be smaller than 'x_max'
    """
    x_min: int = attr.ib(converter=int)
    y_min: int = attr.ib(converter=int)
    x_max: int = attr.ib(converter=int)
    y_max: int = attr.ib(converter=int)
    probability: float = attr.ib(converter=float)
    _np_landmarks: np.ndarray = attr.ib(factory=lambda: np.zeros(shape=(0, 2)),
                                        eq=False)

    @property
    def landmarks(self):
        return self._np_landmarks.astype(int).tolist()

    @x_min.validator
    def check_x_min(self, attribute, value):
        if value > self.x_max:
            raise ValueError("'x_min' must be smaller than 'x_max'")

    @y_min.validator
    def check_y_min(self, attribute, value):
        if value > self.y_max:
            raise ValueError("'y_min' must be smaller than 'y_max'")

    @probability.validator
    def check_probability(self, attribute, value):
        if not (0 <= value <= 1):
            raise ValueError("'probability' must be between 0 and 1")

    @property
    def xy(self):
        return (self.x_min, self.y_min), (self.x_max, self.y_max)

    @property
    def center(self):
        return (self.x_min + self.x_max) // 2, (self.y_min + self.y_max) // 2

    @property
    def width(self):
        return abs(self.x_max - self.x_min)

    @property
    def height(self):
        return abs(self.y_max - self.y_min)

    def similar(self, other: 'BoundingBoxDTO', tolerance: int) -> bool:
        """
        >>> BoundingBoxDTO(50,50,100,100,1).similar(BoundingBoxDTO(50,50,100,100,1),5)
        True
        >>> BoundingBoxDTO(50,50,100,100,1).similar(BoundingBoxDTO(50,50,100,95,1),5)
        True
        >>> BoundingBoxDTO(50,50,100,100,1).similar(BoundingBoxDTO(50,50,100,105,1),5)
        True
        >>> BoundingBoxDTO(50,50,100,100,1).similar(BoundingBoxDTO(50,50,100,94,1),5)
        False
        >>> BoundingBoxDTO(50,50,100,100,1).similar(BoundingBoxDTO(50,50,100,106,1),5)
        False
        """
        return (abs(self.x_min - other.x_min) <= tolerance
                and abs(self.y_min - other.y_min) <= tolerance
                and abs(self.x_max - other.x_max) <= tolerance
                and abs(self.y_max - other.y_max) <= tolerance)

    def similar_to_any(self, others: List['BoundingBoxDTO'], tolerance: int) -> bool:
        """
        >>> BoundingBoxDTO(50,50,100,100,1).similar_to_any([BoundingBoxDTO(50,50,100,105,1),\
                                                            BoundingBoxDTO(50,50,100,106,1)], 5)
        True
        >>> BoundingBoxDTO(50,50,100,100,1).similar_to_any([BoundingBoxDTO(50,50,100,106,1), \
                                                            BoundingBoxDTO(50,50,100,106,1)], 5)
        False
        """
        for other in others:
            if self.similar(other, tolerance):
                return True
        return False

    def is_point_inside(self, xy: Tuple[int, int]) -> bool:
        """
        >>> BoundingBoxDTO(100,700,150,750,1).is_point_inside((125,725))
        True
        >>> BoundingBoxDTO(100,700,150,750,1).is_point_inside((5,5))
        False
        """
        x, y = xy
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def scaled(self, coefficient: float) -> 'BoundingBoxDTO':
        # noinspection PyTypeChecker
        return BoundingBoxDTO(x_min=self.x_min * coefficient,
                              y_min=self.y_min * coefficient,
                              x_max=self.x_max * coefficient,
                              y_max=self.y_max * coefficient,
                              np_landmarks=self._np_landmarks * coefficient,
                              probability=self.probability)

class FaceDetector():
    ml_models = (
        ('retinaface_mnet025_v1', '1ggNFFqpe0abWz6V1A82rnxD6fyxB8W2c'),
    )
    call_counter = 0
    MAX_CALL_COUNTER = 1000
    IMG_LENGTH_LIMIT = 640
    IMAGE_SIZE = 112
    det_prob_threshold = 0.8
    def __init__(self):
        self.model = self._detection_model()

    def _detection_model(self):
        model_file = model_store.find_params_file("models/retinaface_mnet025_v1")
        model = DetectionOnlyFaceAnalysis(model_file)
        model.prepare(ctx_id=-1, nms=0.4)
        return model

    def find_faces(self, img, det_prob_threshold: float = 0.8):
        if det_prob_threshold is None:
            det_prob_threshold = self.det_prob_threshold
        assert 0 <= det_prob_threshold <= 1
        scaler = ImgScaler(self.IMG_LENGTH_LIMIT)
        img = scaler.downscale_img(img)
        results = self.model.get(img, det_thresh=det_prob_threshold)
        boxes =[]
        for result in results:
            downscaled_box_array = result.bbox.astype(np.int).flatten()
            downscaled_box = BoundingBoxDTO(x_min=downscaled_box_array[0],
                                            y_min=downscaled_box_array[1],
                                            x_max=downscaled_box_array[2],
                                            y_max=downscaled_box_array[3],
                                            probability=result.det_score,
                                            np_landmarks=result.landmark)
            box = downscaled_box.scaled(scaler.upscale_coefficient)
            if box.probability <= det_prob_threshold:
                continue
            boxes.append(box)
        return boxes
    
    def crop_face(self, img, box: BoundingBoxDTO):
        return face_align.norm_crop(img, landmark=box._np_landmarks,
                                    image_size=self.IMAGE_SIZE)
    
        # downscaled_box_array = results[0].bbox.astype(np.int).flatten()

        # return downscaled_box_array



def process_images(root_dir: str, output_dir: str, face_detector: FaceDetector):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(root_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root_dir, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue

            faces = face_detector.find_faces(img)
            
            for i, face in enumerate(faces):
                x1, y1, x2, y2 = map(int, face.bbox)
                
                # Ensure coordinates are within image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                
                face_img = img[y1:y2, x1:x2]
                
                output_filename = f"{os.path.splitext(filename)[0]}_face_{i}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                # cv2.imwrite(output_path, face_img)
                print(f"Saved face: {output_path}")



import numpy as np
# should be insightface==0.1.5
# pip install numpy==1.23.5
# pip install mxnet -f https://dist.mxnet.io/python/cpu

from insightface.app import FaceAnalysis
from insightface.model_zoo import (model_store, face_detection,
                                face_recognition) 

class Calculator():
    def __init__(self):
        self.model = self._calculation_model()

    def calc_embedding(self, face_img):
        return self.model.get_embedding(face_img).flatten()


    def _calculation_model(self):
        model_file = model_store.find_params_file("models/model-y1-test2/") #self.get_model_file(self.ml_model) 
        model = face_recognition.FaceRecognition(
            "arcface_mobilefacenet", True, model_file)
        model.prepare(ctx_id=-1)
        return model

def read_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or invalid image format.")
    img = cv2.resize(img, dsize=(112,112))
    return img

def save_embedding(embedding, file_path):
    np.savetxt(file_path, embedding)

# Main function
def main(image_path, output_path):
    # Initialize the calculator
    calculator = Calculator()
    
    # Read the image
    face_img = read_image(image_path)
    
    # Calculate the embedding
    embedding = calculator.calc_embedding(face_img)
    print(embedding)
    
    # Save the embedding to a .txt file
    save_embedding(embedding, output_path)


if __name__ == "__main__":
    model_path = "models/retinaface_mnet025_v1.pth"
    root_dir = "data/NAB_faces/faces"
    output_dir = "data/NAB_faces/faces_cut_1"
    embedding_output_file = "data/NAB_faces/face_embeddings.txt"

    calculator = Calculator()
    
    
    # Calculate the embedding

    face_detector = FaceDetector()
    img = cv2.imread("data/NAB_faces/faces/20240420_080416.jpg")
    boxes = face_detector.find_faces(img)
    for box in boxes:
        face = face_detector.crop_face(img, box)
        # cv2.imshow('hi', face)
        embedding = calculator.calc_embedding(face)
        embedding_array = np.array(embedding)
    
        # Format the array as a string with desired format
        embedding_str = np.array2string(embedding_array, separator=', ', precision=7, floatmode='fixed', suppress_small=True)
    
        # Save the formatted string to a text file
        with open(embedding_output_file, 'w') as f:
            f.write(embedding_str)

    print(embedding)
    cv2.waitKey()
    # process_images(root_dir, output_dir, face_detector)

    # # model_name = 'arcface_mobilefacenet'
    # # file_id = '17TpxpyHuUc1ZTm3RIbfvhnBcZqhyKszV'
    # # downloader = ModelDownloader()
    # # model_path = downloader.download_model(model_name, file_id)

    # input_dir = "data/NAB_faces/faces_cut_1/20240509_091136_face_0.jpg"
   
    # main(image_path=input_dir, output_path=embedding_output_file)
    