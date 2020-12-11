import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import torch
import torch.nn as nn
import os
import torchvision
import utils
import transforms as T
import random

from PIL import Image
from torchvision import datasets
from facenet_pytorch import MTCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.nn import Softmax, Dropout, Linear
from torchvision import models
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor, CenterCrop, Normalize
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
  model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
  hidden_layer = 256
  model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

  return model


H_all, W_all = 248, 173
class ImageEncoder(nn.Module):
  def __init__(self, H_all=H_all, W_all=W_all, num_res=4):
    super(ImageEncoder, self).__init__()
    self.H_all = H_all
    self.W_all = W_all
    self.num_res = num_res

    self.res_block1 = self.make_res_block_(3, 16, 16)
    self.conv1 = nn.Conv2d(3, 16, 1, bias=False)
    self.relu1 = nn.ReLU()
    self.maxpool1 = nn.MaxPool2d(2, 2)

    self.res_block2 = self.make_res_block_(16, 32, 32)
    self.conv2 = nn.Conv2d(16, 32, 1, bias=False)
    self.relu2 = nn.ReLU()
    self.maxpool2 = nn.MaxPool2d(2, 2)

    self.res_block3 = self.make_res_block_(32, 64, 64)
    self.conv3 = nn.Conv2d(32, 64, 1, bias=False)
    self.relu3 = nn.ReLU()
    self.maxpool3 = nn.MaxPool2d(2, 2)

    self.res_block4 = self.make_res_block_(64, 128, 128)
    self.conv4 = nn.Conv2d(64, 128, 1)
    self.relu4 = nn.ReLU()
    self.maxpool4 = nn.MaxPool2d(2, 2)

    self.res_blocks = [(self.res_block1, self.conv1, self.relu1, self.maxpool1),
                       (self.res_block2, self.conv2, self.relu2, self.maxpool2),
                       (self.res_block3, self.conv3, self.relu3, self.maxpool3),
                       (self.res_block4, self.conv4, self.relu4, self.maxpool4)]


  def make_res_block_(self, in_dims, out_dims, num_features):
    return nn.Sequential(
        nn.Conv2d(in_dims, out_dims, 3, bias=False, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dims, out_dims, 3, bias=False, padding=1),
        nn.BatchNorm2d(num_features)
    )
  
  def forward(self, X):
    for i in range(self.num_res):
      X = self.res_block_forward_(X, *self.res_blocks[i])
    return X

  def res_block_forward_(self, X, res_block, conv, activation, pool):
    identity = X
    X = res_block(X)
    return pool(activation(X + conv(identity)))


class CNNRegressor(nn.Module):
  def __init__(self, H_all=H_all, W_all=W_all, num_res=4):
    super(CNNRegressor, self).__init__()
    self.img_encoder = ImageEncoder(H_all, W_all, num_res)
    self.mask_encoder = ImageEncoder(H_all, W_all, num_res)
    H4, W4 = H_all // 16, W_all // 16
    self.reshaper = nn.Conv2d(128, 128, (H4, W4), bias=False)
    self.fc = nn.Sequential(
        nn.Conv2d(128, 16, 1),
        nn.ReLU()
    )
    self.out = nn.Conv2d(16, 2, 1)
  
  def forward(self, img, mask):
    img = self.img_encoder(img)
    mask = self.mask_encoder(mask)
    X = img * mask
    X = self.reshaper(X)
    X = self.fc(X)
    out = self.out(X)
    return out


# based on DCGAN's upsampling part: 
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

class UpSampler(nn.Module):
  def __init__(self, size_grads=[14,28], in_channels=3, 
               num_features_init=8, tgt_size=224):
    # assert tgt_size == size_grads[-1] * 2, "wrong size_grads or tgt_dim"

    super(UpSampler, self).__init__()
    self.size_grads = size_grads
    self.tgt_size = tgt_size
    self.channel_increasers = nn.ModuleList()
    self.layers = nn.ModuleList()

    next_channels = in_channels
    cur_channels = num_features_init

    for _ in range(len(size_grads)):
      self.layers.insert(0, self.make_transpose_conv_block(cur_channels, 
                                                        next_channels))
      self.channel_increasers.insert(0, nn.Conv2d(in_channels, cur_channels, 1, 1, 
                                                  bias=False))
      next_channels = cur_channels
      cur_channels *= 2
    
  
  def make_transpose_conv_block(self, in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
  
  def forward(self, X):
    _, _, H, W = X.shape
    assert H == W, "H is not equal to W"
    if H == self.tgt_size:
      return X
    
    if max(self.size_grads) < 40: # small face model
      ind = self.size_grads.index(H/4)
    else:
      ind = self.size_grads.index(H) # large face model
    X = self.channel_increasers[ind](X)
    for i in range(ind, len(self.layers)):
      X = self.layers[i](X)
    
    return X


class MaskModel(nn.Module):
  def __init__(self, size_grads=[14,28], in_channels=3, 
               num_features_init=8, tgt_size=224):
    super(MaskModel, self).__init__() # tgt_size is the input of the original model, 
                                      # e.g., ResNet, MobileNet, &etc.
    self.upsampler = UpSampler(size_grads=size_grads, in_channels=in_channels, 
                               num_features_init=num_features_init, 
                               tgt_size=tgt_size)
    
    ## TODO: replace the main network below
    # self.resizer = nn.Conv2d(in_channels, in_channels, tgt_size, 1)
    # self.fc = nn.Conv2d(in_channels, 2, 1, 1) # (B, 2, 1, 1)
    # self.out = nn.Softmax(dim=1)
    base_model = mobilemodel
    for param in base_model.parameters():
      param.requires_grad = False
    new_model = nn.Sequential(nn.Linear(in_features=1280, out_features=128),
                       nn.Dropout(p=0.5),
                       nn.Linear(in_features=128,out_features=2))
    base_model.classifier = new_model
    self.classifier = base_model
    # self.out = nn.Softmax(dim=1)
    ## end TODO

  def forward(self, X):
    X = self.upsampler(X)
    # X = self.resizer(X)
    # X = self.fc(X)
    X = self.classifier(X)
    if X.shape[0]!=1:
      X = X.squeeze()
    # X = self.out(X)

    return X

#########################################################################################
# datasets
# reference: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
class PennFudanDataset(torch.utils.data.Dataset):
  def __init__(self, root, transforms):
    self.root = root
    self.transforms = transforms
    self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
    self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
  

  def __getitem__(self, idx):
    img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
    mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path)
    mask = np.array(mask)
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[1:]

    masks = (mask == obj_ids[:, None, None])
    num_objs = len(obj_ids)

    boxes = []
    for i in range(num_objs):
      pos = np.where(masks[i])
      xmin = np.min(pos[1])
      xmax = np.max(pos[1])
      ymin = np.min(pos[0])
      ymax = np.max(pos[0])
      boxes.append([xmin, ymin, xmax, ymax])
    
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.ones((num_objs,), dtype=torch.int64)
    masks = torch.as_tensor(masks, dtype=torch.uint8)
    image_id = torch.tensor([idx])
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    iscrowd = torch.zeros((num_objs, ), dtype=torch.int64)

    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["masks"] = masks
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd

    if self.transforms is not None:
      img, target = self.transforms(img, target)  # NOTE: not from torchvision.transforms, 
                                                  # but from another library
    return img, target
  
  def __len__(self):
    return len(self.imgs)

  
class FaceDataset(Dataset):
  def __init__(self, imgs, masks, locs):
    self.imgs = imgs.astype(np.float32)
    self.masks = masks.astype(np.float32)
    self.locs = locs.astype(np.float32)
  
  def __len__(self):
    return self.imgs.shape[0]
  
  def __getitem__(self, idx):
    # dict_out = {"mask": torch.tensor(self.masks[idx]), "locs": (self.locs[idx, 0], self.locs[idx, 1])}
    # return torch.tensor(self.imgs[idx], dtype=torch.float32), torch.tensor(self.masks[idx], dtype=torch.float32), \
    # (self.locs[idx, 0], self.locs[idx, 1])
    return torch.tensor(self.imgs[idx]), torch.tensor(self.masks[idx]), \
    (self.locs[idx, 0], self.locs[idx, 1])

  
# masked: 1, unmasked: 0
class ResizedDataset(Dataset):
  def __init__(self, masked_paths: list, unmasked_paths: list, tgt_size: tuple):
    self.masked_paths = masked_paths
    self.unmasked_paths = unmasked_paths
    self.transform = T.Compose([
      T.ToPILImage(),
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      T.Resize(tgt_size)
    ])
  
  def __len__(self):
    return len(self.masked_paths) + len(self.unmasked_paths)
  
  def __getitem__(self, idx):
    num_masked, num_unmasked = len(self.masked_paths), len(self.unmasked_paths)
    assert idx < (num_masked + num_unmasked) and idx >= 0, "invalid index"
    path, label = None, -1
    if idx >= num_masked:
      idx -= num_masked
      path = self.unmasked_paths[idx]
      label = 0
    else:
      path = self.masked_paths[idx]
      label = 1
    
    # img = (cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB) / 255).astype(np.float32)
    img = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)

    return self.transform(img), label
