import sys # nopep8
sys.path.append("") # nopep8

import torch
import numpy as np
import os
import wandb 
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from  torchvision.transforms import InterpolationMode , v2
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training

from src.Utils.utils import *

data_dir = 'data/facenet_vn_cropped'
facenet_config_path = 'config/facenet.yaml'
facenet_config = read_config(path = facenet_config_path)
EPOCHS = facenet_config['EPOCHS']
PATIENCE = facenet_config['PATIENCE']
BATCH_SIZE = facenet_config['BATCH_SIZE']
learning_rate = facenet_config['learning_rate']
IMG_SIZE = facenet_config['IMG_SIZE']
RANDOM_SEED = facenet_config['RANDOM_SEED']
NUM_CLASSES = facenet_config['NUM_CLASSES']
WEIGHT_DECAY = facenet_config['WEIGHT_DECAY']
LR_WARMUP = facenet_config['LR_WARMUP']
CLIP_GRAD_NORM = facenet_config['CLIP_GRAD_NORM']
PRETRAINED_MODEL = facenet_config['PRETRAINED_MODEL']
MODEL_DIR = facenet_config['MODEL_DIR']
MODEL_DIR = rename_model(model_dir = MODEL_DIR, prefix='facenet')
facenet_config['MODEL_DIR'] = MODEL_DIR


# wandb.init(
#     # set the wandb project where this run will be logged
#     project="facenet_tune",
#     name= os.path.splitext(os.path.basename(MODEL_DIR))[0],
#     notes=facenet_config['notes'],
#     tags=facenet_config['tags'],
#     # track hyperparameters and run metadata
#     config=facenet_config
# )

EPOCH_LEN = len(str(EPOCHS))
torch.manual_seed(RANDOM_SEED)

NUM_WORKERS = 0 if os.name == 'nt' else 8

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


trans = v2.Compose([
    v2.Resize(232, interpolation=InterpolationMode.BICUBIC,),
    v2.CenterCrop(IMG_SIZE),
    v2.ToTensor(),
    # v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    fixed_image_standardization,
])

dataset = datasets.ImageFolder(data_dir, transform=trans)

loader = DataLoader(
    dataset,
    num_workers=NUM_WORKERS,
    batch_size=BATCH_SIZE,
    collate_fn=training.collate_pil
)

model = InceptionResnetV1(
    classify=True,
    pretrained=PRETRAINED_MODEL,
    num_classes=len(dataset.class_to_idx)
).to(device)


criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = WEIGHT_DECAY)
scheduler_linear = lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=LR_WARMUP)
scheduler_cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=490, eta_min=learning_rate/100)
scheduler_lr = lr_scheduler.SequentialLR(optimizer, [scheduler_linear,scheduler_cosine],milestones=[10])

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor



# trans = transforms.Compose([
#     np.float32,
#     transforms.ToTensor(),
#     fixed_image_standardization
# ])


img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.8 * len(img_inds))]
val_inds = img_inds[int(0.8 * len(img_inds)):]

train_loader = DataLoader(
    dataset,
    num_workers=NUM_WORKERS,
    batch_size=BATCH_SIZE,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    dataset,
    num_workers=NUM_WORKERS,
    batch_size=BATCH_SIZE,
    sampler=SubsetRandomSampler(val_inds)
)

loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}

print('\n\nInitial')
print('-' * 10)

for epoch in range(EPOCHS):
    print('\nEpoch {}/{}'.format(epoch + 1, EPOCHS))
    print('-' * 10)

    model.train()
    training.pass_epoch(
        model, loss_fn, train_loader, optimizer, scheduler_lr,
        batch_metrics=metrics, show_running=True, device=device,
        # writer=writer
    )

    model.eval()
    training.pass_epoch(
        model, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        # writer=writer
    )

model.eval()
# Example input (dummy input to export the model)
dummy_input = torch.randn(1, 3, 224, 224)  # Example input (1 batch of 3-channel 224x224 images)

root_dir ="models/facenet_tune/"
# Export the PyTorch model to ONNX
onnx_path = root_dir+ 'resnet_model.onnx'
torch.onnx.export(model, dummy_input, onnx_path, verbose=True)

# Convert ONNX model to TensorFlow's .pb format using tf.compat.v1
import tensorflow as tf
import onnx

# Load ONNX model
onnx_model = onnx.load(onnx_path)

# Convert ONNX model to TensorFlow format
tf_rep = tf.compat.v1.prepare(onnx_model)

# Save TensorFlow model in .pb format
tf_pb_path = root_dir + 'model_19_6.pb'
tf_rep.export_graph(tf_pb_path)


