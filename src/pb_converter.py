import sys
sys.path.append("")

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import datasets, transforms

import torch
import torch.onnx
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import tensorflow.compat.v1 as tf1

import onnxruntime

from src.Utils.utils import *
from src.facenet_triplet.utils import *

import onnx
from onnx import helper
import math



def convert_pytorch_to_onnx(model, input_shape, onnx_path):
    # Set the model to evaluation mode
    model.eval()

    # Create a dummy input tensor
    dummy_input = torch.randn(input_shape, device=device)

    # Export the model to ONNX
    torch.onnx.export(model, 
                      dummy_input, 
                      onnx_path, 
                      export_params=True, 
                      opset_version=15, 
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['embeddings'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'embeddings': {0: 'batch_size'}})
    

    print(f"PyTorch model exported to ONNX: {onnx_path}")

def convert_onnx_to_tf(onnx_path, tf_path):
    # Load the ONNX model
    import onnx2tf
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=tf_path,
        output_tfv1_pb=True,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=False,
    )
    
    print(f"ONNX model converted to TensorFlow: {tf_path}")


def load_image(image_path):
    # Define the transformations for the input image
    transform = transforms.Compose([
        transforms.Resize((160, 160)),  # Assuming the model expects 160x160 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # Load the image and apply transformations
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    # Add batch dimension
    image = image.unsqueeze(0)
    return image

def get_pytorch_embeddings(model, image_path, device):
    model = model.to(device)
    model.eval()
    image = load_image(image_path).to(device)
    with torch.no_grad():
        embeddings = model.get_embedding(image)
    return embeddings.cpu().numpy()

def get_onnx_embeddings(onnx_model_path, image_path):
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    image = load_image(image_path).numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0]

def compare_embeddings(embeddings1, embeddings2, tolerance=1e-3):
    return np.allclose(embeddings1, embeddings2, atol=tolerance)

def check_models_output(pytorch_model, onnx_model_path, image_path, device):
    pytorch_embeddings = get_pytorch_embeddings(pytorch_model, image_path, device)
    onnx_embeddings = get_onnx_embeddings(onnx_model_path, image_path)
    are_equal = compare_embeddings(pytorch_embeddings, onnx_embeddings)
    return pytorch_embeddings, onnx_embeddings, are_equal

def load_frozen_graph(frozen_graph_filename):
    # with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
    #     graph_def = tf.compat.v1.GraphDef()
    #     graph_def.ParseFromString(f.read())

    # with tf.compat.v1.Graph().as_default() as graph:
    #     tf.import_graph_def(graph_def, name="")

    with tf1.Graph().as_default() as graph:
        graph_def = tf1.GraphDef()
        from tensorflow.python.platform import gfile

        with gfile.FastGFile(frozen_graph_filename, 'rb') as f:
            model = f.read()
        graph_def.ParseFromString(model)
        tf1.import_graph_def(graph_def, name='')

    return graph

def load_and_preprocess_tf_image(image_path, target_size=(160, 160)):
    # Load the image
    img = Image.open(image_path)
    
    # Resize the image
    img = img.resize(target_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(img).astype(np.float32)
    img_array = (img_array - 127.5) / 128.0  # Normalize to [-1, 1]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_tf_embeddings(graph, image_path):
    # Load and preprocess the image
    image = load_and_preprocess_tf_image(image_path)
    
    # Get input and output tensors
    input_tensor = graph.get_tensor_by_name("input:0")
    
    # Try different possible output tensor names
    possible_output_names = ["embeddings", "Identity", "output"]
    output_tensor = None
    
    for name in possible_output_names:
        try:
            output_tensor = graph.get_tensor_by_name(f"{name}:0")
            print(f"Output tensor found: {name}")
            break
        except KeyError:
            continue
    
    if output_tensor is None:
        raise ValueError("Could not find the output tensor. Please check the graph structure.")
    
    try:
        phase_train_tensor = graph.get_tensor_by_name("phase_train:0")
        feed_dict = {input_tensor: image, phase_train_tensor: False}
    except KeyError:
        # If phase_train is not found, proceed without it
        feed_dict = {input_tensor: image}
    
    # Run the session to get embeddings
    with tf.compat.v1.Session(graph=graph) as sess:
        embeddings = sess.run(output_tensor, feed_dict=feed_dict)
    
    return embeddings

def check_tf_onnx_output(tf_model_path, onnx_model_path, image_path):
    tf_graph = load_frozen_graph(tf_model_path)
    tf_embeddings = get_tf_embeddings(tf_graph, image_path)
    onnx_embeddings = get_onnx_embeddings(onnx_model_path, image_path)
    are_equal = compare_embeddings(tf_embeddings, onnx_embeddings)
    return tf_embeddings, onnx_embeddings, are_equal

def prewhiten(img):
    """ Normalize image."""
    mean = np.mean(img)
    std = np.std(img)
    std_adj = np.maximum(std, 1.0 / np.sqrt(img.size))
    y = np.multiply(np.subtract(img, mean), 1 / std_adj)
    return y

# import tf1
# def embedding_calculator(ml_model_file):
#         with tf1.Graph().as_default() as graph:
#             graph_def = tf1.GraphDef()
#             with gfile.FastGFile(ml_model_file, 'rb') as f:
#                 model = f.read()
#             graph_def.ParseFromString(model)
#             tf1.import_graph_def(graph_def, name='')
#             return _EmbeddingCalculator(graph=graph, sess=tf1.Session(graph=graph))
        
# def calculate_embeddings(cropped_images):
#     """Run forward pass to calculate embeddings"""
#     prewhitened_images = [prewhiten(img) for img in cropped_images]
#     calc_model = embedding_calculator(tf_model_path)
#     graph_images_placeholder = calc_model.graph.get_tensor_by_name("input:0")

#     possible_output_names = ["embeddings", "Identity", "output"]
#     graph_embeddings = None

#     for name in possible_output_names:
#         try:
#             graph_embeddings = calc_model.graph.get_tensor_by_name(f"{name}:0")
#             print(f"Output tensor found: {name}")
#             break
#         except KeyError:
#             continue

#     try:
#         graph_phase_train_placeholder = calc_model.graph.get_tensor_by_name("phase_train:0")
#     except KeyError:
#         graph_phase_train_placeholder = None

#     # graph_embeddings = calc_model.graph.get_tensor_by_name("embeddings:0")
#     # graph_phase_train_placeholder = calc_model.graph.get_tensor_by_name("phase_train:0")
#     embedding_size = graph_embeddings.get_shape()[1]
#     image_count = len(prewhitened_images)
#     batches_per_epoch = int(math.ceil(1.0 * image_count / 25))
#     embeddings = np.zeros((image_count, embedding_size))
#     for i in range(batches_per_epoch):
#         start_index = i * 25
#         end_index = min((i + 1) * 25, image_count)
#         if graph_phase_train_placeholder:
#             feed_dict = {graph_images_placeholder: prewhitened_images, graph_phase_train_placeholder: False}
#         else:
#             feed_dict = {graph_images_placeholder: prewhitened_images}

#         embeddings[start_index:end_index, :] = calc_model.sess.run(
#             graph_embeddings, feed_dict=feed_dict)
#     return embeddings

# res_model = InceptionResnetV1(pretrained=PRETRAINED_MODEL, classify=False,
#                                num_classes=None, device=device)
# facenet_embedding_net = FacenetEmbeddingNet(res_model)

# facenet_embedding_net.load_state_dict(torch.load("models/facenet_tune/facenet_2024_07_17_4.pth"))
# facenet_embedding_net = torch.load("models/facenet_tune/facenet_2024_07_18_9.pth")
# facenet_embedding_net = facenet_embedding_net.to(device)

# Load your PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
facenet_embedding_net = torch.load("models/facenet_tune/facenet_2024_07_18_9.pth", map_location=device)
facenet_embedding_net = facenet_embedding_net.to(device)

# Set the model to evaluation mode
facenet_embedding_net.eval()

# Define the input shape
batch_size = 1
channels = 3
height = 160  # Adjust based on your model's input size
width = 160   # Adjust based on your model's input size
input_shape = (batch_size, channels, height, width)

# Define paths
onnx_path = "models/facenet_embedding_net.onnx"
tf_path = "models/facenet_embedding_net_tf"
frozen_graph_path = "models/facenet_embedding_net_frozen.pb"

# Convert PyTorch to ONNX
convert_pytorch_to_onnx(facenet_embedding_net, input_shape, onnx_path)


# Check if the outputs are the same
image_path = "data/NAB_faces_cropped/faces/20240509_091139.jpg"  # Path to the image to test
pytorch_embeddings, onnx_embeddings, are_outputs_same = check_models_output (facenet_embedding_net, onnx_path, 
                                                                            image_path, device)

print(f"Are the outputs of the PyTorch model and the ONNX model the same? {are_outputs_same}")

# Convert ONNX to TensorFlow
# convert_onnx_to_tf(onnx_path, tf_path)

tf_model_path = "models/20180402-114758/20180402-114758.pb"
# tf_model_path = "models/facenet_embedding_net_tf/facenet_embedding_net_float32.pb"
tf_embeddings, onnx_embeddings, are_outputs_same = check_tf_onnx_output(tf_model_path, onnx_path, image_path)

print(f"Are the outputs of the TensorFlow and ONNX models the same? {are_outputs_same}")