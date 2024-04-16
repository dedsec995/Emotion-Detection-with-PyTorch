import os, csv
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchsummary import summary
import torch.nn.functional as F
from torch.utils.mobile_optimizer import optimize_for_mobile

mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 7)


model = mobilenet
model.load_state_dict(torch.load("checkpoint_epoch_30.pt"))
torchscript_model = torch.jit.script(model)
torchscript_model_optimized = optimize_for_mobile(torchscript_model)
torch.jit.save(torchscript_model_optimized, "mobilenet_v2_lite.ptl")

torchscript_model_optimized._save_for_lite_interpreter("mobilenet_v2_lite2.ptl")
