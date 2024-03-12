from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from src.Utils.utils import timeit

@timeit
def liveness_predict(face, liveness_model, liveness_label_encoder):
    '''face: the RGB image of face'''

    face = cv2.resize(face, (32, 32))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    preds = liveness_model.predict(face)[0]
    j = np.argmax(preds)
    label = liveness_label_encoder.classes_[j]
    return preds, label
