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

import wandb
from torchvision import datasets
from torchvision.transforms import InterpolationMode , v2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set up data loaders
from src.facenet_triplet.datasets import TripletFace, BalancedBatchSampler, AugmentedImageFolder
from src.facenet_triplet.networks import TripletNet, FacenetEmbeddingNet, EmbeddingNet
from src.facenet_triplet.losses import TripletLoss, OnlineContrastiveLoss, OnlineTripletLoss
from src.facenet_triplet.trainer import fit, EarlyStopping
from src.facenet_triplet.metrics import AverageNonzeroTripletsMetric

device = "cuda" if torch.cuda.is_available() else "cpu"
cuda = torch.cuda.is_available()


facenet_config_path = 'config/facenet.yaml'
facenet_config = read_config(path = facenet_config_path)
EPOCHS = facenet_config['EPOCHS']
PATIENCE = facenet_config['PATIENCE']
BATCH_SIZE = facenet_config['BATCH_SIZE']
IMG_SIZE = facenet_config['IMG_SIZE']
RANDOM_SEED = facenet_config['RANDOM_SEED']
LR_WARMUP = facenet_config['LR_WARMUP']
MODEL_DIR = facenet_config['MODEL_DIR']
PIN_MEMORY = facenet_config['PIN_MEMORY']
IMG_SIZE = facenet_config['IMG_SIZE']

CLIP_GRAD_NORM = facenet_config['CLIP_GRAD_NORM']
PRETRAINED_MODEL = facenet_config['PRETRAINED_MODEL']

log_interval = facenet_config['log_interval']
learning_rate = facenet_config['learning_rate']
margin = facenet_config['margin']

n_samples = facenet_config['n_samples']
n_classes =int(BATCH_SIZE/n_samples) 

MODEL_DIR = rename_model(model_dir = MODEL_DIR, prefix='facenet')
facenet_config['MODEL_DIR'] = MODEL_DIR
NUM_WORKERS = 0 if os.name == 'nt' else 8

train_dir = facenet_config['train_dir']
test_dir = facenet_config['test_dir']

wandb.init(
    # set the wandb project where this run will be logged
    project="facenet",
    name= os.path.splitext(os.path.basename(MODEL_DIR))[0],
    notes=facenet_config['notes'],
    tags=facenet_config['tags'],
    # track hyperparameters and run metadata
    config=facenet_config
)


num_classes_total = count_folders(train_dir)

wandb.log({
    "n_samples": n_samples,
    "NUM_WORKERS": NUM_WORKERS,
    "num_classes_total": num_classes_total,
})

# wandb = None 

transform_original = v2.Compose([
    v2.Resize(IMG_SIZE, interpolation=InterpolationMode.BICUBIC,),
    v2.CenterCrop(IMG_SIZE),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


albu_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),  # Random horizontal flip
    A.RandomResizedCrop(height=160, width=160, scale=(0.9, 1.0), ratio=(0.9, 1.1), p=0.5),  # Small crop to 155 and resize back to 160
    A.Rotate(limit=15, p=0.5),  # Random rotation between -15 and 15 degrees
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalization
    ToTensorV2(),  # Convert image to tensor
])

    
# train_face_dataset = datasets.ImageFolder(train_dir, transform=transform_original)
train_face_dataset = AugmentedImageFolder(train_dir, transform=transform_original, albu_transform=albu_transforms)
test_face_dataset = datasets.ImageFolder(test_dir, transform=transform_original)
train_face_dataset.train =True

wandb.log({
    "train_face_dataset": len(train_face_dataset),
    "test_face_dataset": len(test_face_dataset),
})

print(f"train_face_dataset: {len(train_face_dataset)}")
print(f"test_face_dataset: {len(test_face_dataset)}")


res_model = InceptionResnetV1(pretrained=PRETRAINED_MODEL, classify=False,
                               num_classes=None, device=device)
facenet_embedding_net = FacenetEmbeddingNet(res_model)
facenet_embedding_net = facenet_embedding_net.to(device)
facenet_model = TripletNet(facenet_embedding_net)
facenet_model = facenet_model.to(device)

# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class

train_batch_sampler = BalancedBatchSampler(train_face_dataset, n_classes=n_classes, n_samples=n_samples, is_dataset=True)
test_batch_sampler = BalancedBatchSampler(test_face_dataset, n_classes=n_classes, n_samples=n_samples, is_dataset=True)


kwargs = {'num_workers': NUM_WORKERS, 'pin_memory': PIN_MEMORY} if cuda else {}

online_train_loader = torch.utils.data.DataLoader(train_face_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_face_dataset, batch_sampler=test_batch_sampler, **kwargs)


# loss_fn = OnlineContrastiveLoss(margin, HardNegativePairSelector()) # Online Pair selection
loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin)) # Online Triplet selection

optimizer = optim.Adam(facenet_model.parameters(), lr=learning_rate)
scheduler_linear = lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=LR_WARMUP)
scheduler_cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=490, eta_min=learning_rate/100)
scheduler = lr_scheduler.SequentialLR(optimizer, [scheduler_linear,scheduler_cosine],milestones=[10])

fit(train_loader = online_train_loader, 
                val_loader=online_test_loader, 
                model= facenet_embedding_net, 
                model_path=MODEL_DIR,
                patience=PATIENCE,
                loss_fn=loss_fn,
                optimizer=optimizer, 
                scheduler = scheduler, 
                n_epochs=EPOCHS, 
                device=device, 
                log_interval=log_interval,
                wandb = wandb,
                metrics=[AverageNonzeroTripletsMetric()]
                )

# facenet_model = torch.load("models/facenet_tune/facenet_2024_07_15_1.pth")

# plot_model_result(facenet_model, triplet_train_face_loader, device)


