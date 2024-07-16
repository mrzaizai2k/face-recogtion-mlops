import sys # nopep8
sys.path.append("") # nopep8


import os
import torch
from torchinfo import summary
# from torchvision import transforms as T
from src.Utils.utils import *
from src.facenet_triplet.utils import *
from facenet_pytorch import InceptionResnetV1

import torch
from torch.optim import lr_scheduler
import torch.optim as optim

from src.facenet_triplet.trainer import fit

from torchvision import datasets
from torchvision.transforms import InterpolationMode , v2

from src.facenet_triplet.metrics import AverageNonzeroTripletsMetric

# Set up data loaders
from src.facenet_triplet.datasets import TripletFace, BalancedBatchSampler
from src.facenet_triplet.networks import TripletNet, FacenetEmbeddingNet, EmbeddingNet
from src.facenet_triplet.losses import TripletLoss, OnlineContrastiveLoss


device = "cuda" if torch.cuda.is_available() else "cpu"
cuda = torch.cuda.is_available()


facenet_config_path = 'config/facenet.yaml'
facenet_config = read_config(path = facenet_config_path)
EPOCHS = facenet_config['EPOCHS']
PATIENCE = facenet_config['PATIENCE']
BATCH_SIZE = facenet_config['BATCH_SIZE']
IMG_SIZE = facenet_config['IMG_SIZE']
RANDOM_SEED = facenet_config['RANDOM_SEED']
WEIGHT_DECAY = facenet_config['WEIGHT_DECAY']
LR_WARMUP = facenet_config['LR_WARMUP']
CLIP_GRAD_NORM = facenet_config['CLIP_GRAD_NORM']
PRETRAINED_MODEL = facenet_config['PRETRAINED_MODEL']
MODEL_DIR = facenet_config['MODEL_DIR']
PIN_MEMORY = facenet_config['PIN_MEMORY']
IMG_SIZE = facenet_config['IMG_SIZE']

log_interval = facenet_config['log_interval']
learning_rate = facenet_config['learning_rate']
margin = facenet_config['margin']


MODEL_DIR = rename_model(model_dir = MODEL_DIR, prefix='facenet')
facenet_config['MODEL_DIR'] = MODEL_DIR
NUM_WORKERS = 0 if os.name == 'nt' else 8


output_dir =  "models/facenet_tune"
train_dir = 'data/crop_3'
test_dir = 'data/facenet_vn_cropped_test'

def count_folders(path):
    """ Count the number of folders in the given directory """
    return len([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])


num_classes = count_folders(train_dir)
print(f'Number of classes: {num_classes}')


transform_original = v2.Compose([
    v2.Resize(IMG_SIZE, interpolation=InterpolationMode.BICUBIC,),
    v2.CenterCrop(IMG_SIZE),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
train_face_dataset = datasets.ImageFolder(train_dir, transform=transform_original)
test_face_dataset = datasets.ImageFolder(test_dir, transform=transform_original)
train_face_dataset.train =True


res_model = InceptionResnetV1(pretrained=PRETRAINED_MODEL, classify=False,
                               num_classes=None, device=device)
facenet_embedding_net = FacenetEmbeddingNet(res_model)
facenet_embedding_net = facenet_embedding_net.to(device)
facenet_model = TripletNet(facenet_embedding_net)
facenet_model = facenet_model.to(device)

triplet_train_face_dataset = TripletFace(train_face_dataset, random_seed=RANDOM_SEED) # Returns triplets of images
triplet_test_face_dataset = TripletFace(test_face_dataset, random_seed=RANDOM_SEED) # Returns triplets of images

# plot_triplet(triplet_train_face_dataset[10][0])

kwargs = {'num_workers': NUM_WORKERS, 'pin_memory': PIN_MEMORY} if cuda else {}

triplet_train_face_loader = torch.utils.data.DataLoader(triplet_train_face_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
triplet_test_face_loader = torch.utils.data.DataLoader(triplet_test_face_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)


loss_fn = TripletLoss(margin)

optimizer = optim.Adam(facenet_model.parameters(), lr=learning_rate)
scheduler_linear = lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=LR_WARMUP)
scheduler_cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=490, eta_min=learning_rate/100)
scheduler = lr_scheduler.SequentialLR(optimizer, [scheduler_linear,scheduler_cosine],milestones=[10])


# fit(train_loader = triplet_train_face_loader, 
#     val_loader=triplet_test_face_loader, 
#     model= facenet_model, 
#     loss_fn=loss_fn,
#     optimizer=optimizer, 
#     scheduler = scheduler, 
#     n_epochs=EPOCHS, 
#     device=device, 
#     log_interval=log_interval,)


# torch.save(facenet_model.state_dict(), "models/facenet_tune/facenet.pt")
# torch.save(facenet_model, MODEL_DIR)


# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
n_classes = 4
n_samples =int(BATCH_SIZE/n_classes) 

train_batch_sampler = BalancedBatchSampler(train_face_dataset, n_classes=n_classes, n_samples=n_samples, is_dataset=True)
test_batch_sampler = BalancedBatchSampler(test_face_dataset, n_classes=n_classes, n_samples=n_samples, is_dataset=True)


kwargs = {'num_workers': NUM_WORKERS, 'pin_memory': PIN_MEMORY} if cuda else {}

online_train_loader = torch.utils.data.DataLoader(train_face_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_face_dataset, batch_sampler=test_batch_sampler, **kwargs)


loss_fn = OnlineContrastiveLoss(margin, HardNegativePairSelector())
optimizer = optim.Adam(facenet_model.parameters(), lr=learning_rate)
scheduler_linear = lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=LR_WARMUP)
scheduler_cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=490, eta_min=learning_rate/100)
scheduler = lr_scheduler.SequentialLR(optimizer, [scheduler_linear,scheduler_cosine],milestones=[10])



fit(train_loader = online_train_loader, 
                val_loader=online_test_loader, 
                model= facenet_embedding_net, 
                loss_fn=loss_fn,
                optimizer=optimizer, 
                scheduler = scheduler, 
                n_epochs=EPOCHS, 
                device=device, 
                log_interval=log_interval,)

# facenet_model = torch.load("models/facenet_tune/facenet_2024_07_15_1.pth")

# plot_model_result(facenet_model, triplet_train_face_loader, device)


