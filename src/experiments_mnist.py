import sys
sys.path.append("")

import os
import random
import requests
from tqdm import tqdm
from PIL import Image

# Data analytics and visualisations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# PyTorch packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from torchinfo import summary
from torchvision import transforms as T
from src.Utils.utils import *
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training

import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from src.siamese_triplet.trainer import fit
cuda = torch.cuda.is_available()

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from  torchvision.transforms import InterpolationMode 
from torchvision.transforms import v2
from torchvision import datasets


# Set up data loaders
from src.siamese_triplet.datasets import TripletMNIST
from src.siamese_triplet.networks import EmbeddingNet, TripletNet
from src.siamese_triplet.losses import TripletLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
device


data_dir = 'data/facenet_vn_cropped'
facenet_config_path = 'config/facenet.yaml'
facenet_config = read_config(path = facenet_config_path)
EPOCHS = facenet_config['EPOCHS']
PATIENCE = facenet_config['PATIENCE']
BATCH_SIZE = facenet_config['BATCH_SIZE']
learning_rate = facenet_config['learning_rate']
IMG_SIZE = facenet_config['IMG_SIZE']
RANDOM_SEED = facenet_config['RANDOM_SEED']
WEIGHT_DECAY = facenet_config['WEIGHT_DECAY']
LR_WARMUP = facenet_config['LR_WARMUP']
CLIP_GRAD_NORM = facenet_config['CLIP_GRAD_NORM']
PRETRAINED_MODEL = facenet_config['PRETRAINED_MODEL']
MODEL_DIR = facenet_config['MODEL_DIR']
MODEL_DIR = rename_model(model_dir = MODEL_DIR, prefix='facenet')
facenet_config['MODEL_DIR'] = MODEL_DIR
NUM_WORKERS = 0 if os.name == 'nt' else 8


data_dir="data/facenet_vn_cropped"

mean, std = 0.1307, 0.3081

train_dataset = MNIST('../data/MNIST', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean,), (std,))
                             ]))
test_dataset = MNIST('../data/MNIST', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                            ]))
n_classes = 10


transform_original = v2.Compose([
    v2.Resize(160, interpolation=InterpolationMode.BICUBIC,),
    v2.CenterCrop(160),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
train_face_dataset = datasets.ImageFolder(data_dir, transform=transform_original)
test_face_dataset = datasets.ImageFolder("data/facenet_vn_cropped_test", transform=transform_original)
train_face_dataset.train =True


class TripletFace(Dataset):
    """
    Train: For each sample (anchor), randomly chooses a positive and negative sample.
    Test: Creates fixed triplets for testing.
    """

    def __init__(self, image_folder_dataset, train=True):
        self.image_folder_dataset = image_folder_dataset
        self.train = train
        self.transform = self.image_folder_dataset.transform

        # Split dataset into train and test
        if self.train:
            self.data = [self.image_folder_dataset.samples[i][0] for i in range(len(self.image_folder_dataset))]
            self.labels = [self.image_folder_dataset.samples[i][1] for i in range(len(self.image_folder_dataset))]
            self.labels_set = set(self.labels)
            self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                     for label in self.labels_set}
        else:
            self.data = [self.image_folder_dataset.samples[i][0] for i in range(len(self.image_folder_dataset))]
            self.labels = [self.image_folder_dataset.samples[i][1] for i in range(len(self.image_folder_dataset))]
            self.labels_set = set(self.labels)
            self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.labels[i]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.labels[i]]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1_path, label1 = self.data[index], self.labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            img2_path = self.data[positive_index]
            img3_path = self.data[negative_index]
        else:
            img1_path = self.data[self.test_triplets[index][0]]
            img2_path = self.data[self.test_triplets[index][1]]
            img3_path = self.data[self.test_triplets[index][2]]

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        img3 = Image.open(img3_path).convert('RGB')
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return (img1, img2, img3), []

    def __len__(self):
        return len(self.image_folder_dataset)

def plot_triplet(images):
    # Plot the images
    fig, ax = plt.subplots(1, 3)
    labels = ["Anchor", "Positive", "Negative"]
    for i, img in enumerate(images):
        print(img.shape)
        # The shape of the image is changed to (3, 112, 112),
        # for plotting, we need the shape to be (112, 112, 3)
        ax[i].imshow(img.permute(1,2,0))

        # Hide the ticks
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        
        # Add a label
        ax[i].set_title(labels[i])

class FacenetEmbeddingNet(nn.Module):
    def __init__(self, facenet_model):
        super(FacenetEmbeddingNet, self).__init__()
        self.facenet_model = facenet_model

    def forward(self, x):
        # Forward pass through Facenet model
        return self.facenet_model(x)

    def get_embedding(self, x):
        # Get embeddings using the Facenet model
        return self.facenet_model(x)

# Example usage:
# Assuming 'device' is defined and set appropriately
res_model = InceptionResnetV1(pretrained='vggface2', classify=False, num_classes=None, device=device)
facenet_embedding_net = FacenetEmbeddingNet(res_model)
facenet_model = TripletNet(facenet_embedding_net)

triplet_train_face_dataset = TripletFace(train_face_dataset) # Returns triplets of images
triplet_test_face_dataset = TripletFace(test_face_dataset) # Returns triplets of images
# plot_triplet(triplet_train_face_dataset[100][0])

batch_size = 4
kwargs = {'num_workers': NUM_WORKERS, 'pin_memory': True} if cuda else {}
triplet_train_face_loader = torch.utils.data.DataLoader(triplet_train_face_dataset, batch_size=batch_size, shuffle=True, **kwargs)
triplet_test_face_loader = torch.utils.data.DataLoader(triplet_test_face_dataset, batch_size=batch_size, shuffle=False, **kwargs)

margin = 1.
if cuda:
    facenet_model.cuda()
loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(facenet_model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 1
log_interval = 10

fit(triplet_train_face_loader, triplet_test_face_loader, facenet_model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)


triplet_train_dataset = TripletMNIST(train_dataset) # Returns triplets of images
triplet_test_dataset = TripletMNIST(test_dataset)
batch_size = 32
kwargs = {'num_workers': NUM_WORKERS, 'pin_memory': True} if cuda else {}
triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

margin = 1.
embedding_net = EmbeddingNet()
model = TripletNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 100

fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)



