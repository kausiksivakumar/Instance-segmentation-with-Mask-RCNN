import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.detection.image_list import ImageList
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import figure
import h5py
import pdb
from functools import partial
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import random_split, DataLoader 
import torchvision.transforms as transforms
from sklearn import metrics
from torchvision.ops import box_iou
from torchvision.ops import batched_nms
import torch
from torchvision import transforms
from torch.nn import functional as F
from torch import nn, Tensor
import torchvision
import gc
import time

'''
Dataset and datamodule 
'''
class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
      #############################################
      # Initialize  Dataset
      #############################################
      '''
      Requirements path = [img_path, mask_path, label_path, bbx_path]
      '''
      self.bboxes  =   np.load(path[3],allow_pickle=True)
      self.labels  =   np.load(path[2],allow_pickle=True)
      self.images  =   self.read_h5py(path[0])
      self.masks   =   self.read_h5py(path[1])

      # Scaling params
      self.x_scale        = 800/300
      self.y_scale        = 1066/400
      self.x_plus         = 11

      #mask_idx for lookup -- fn in utils.py
      self.mask_idx       = self.mask_idx(self.images,self.labels)

      # Transforms
      self.transform             = transforms.Compose([ transforms.Resize((800,1066)),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225]),
                                             transforms.Pad((11,0)) ])

      self.mask_transform        = transforms.Compose([ transforms.Resize((800,1066)),
                                             transforms.Pad((11,0))
                                             ])
    def read_h5py(self,name):
      '''
      Reads .h5 files and return np array
      '''
      with h5py.File(name, "r") as f:
        a_group_key = list(f.keys())[0]
        # ds_obj = f[a_group_key]      # returns as a h5py dataset object
        ds_arr = f[a_group_key][()]  # returns as a numpy array
      
      return ds_arr

    def mask_idx(self,images_orig,labels_orig):
      # process labels to get the idx for mask 
      # as mask is flattened
      process_label_num       = {}
      j = 0
      for i in range(len(images_orig)):
        process_label_num[i] = j
        j = j+ len(labels_orig[i])

      return process_label_num

    def __getitem__(self, idx):
        ################################
        # return transformed images,labels,masks,boxes,index
        ################################
        transed_img     = (torch.from_numpy(self.images[idx].copy().astype(float))/255).float()
        label           = torch.from_numpy(self.labels[idx].copy()).float()
        transed_bbox    = torch.from_numpy(self.bboxes[idx].copy()).float()
        
        # Accounting for scaling and padding for the bounding box
        transed_bbox[:,0]       *=  self.x_scale
        transed_bbox[:,0]       +=  self.x_plus

        transed_bbox[:,2]       *=  self.x_scale
        transed_bbox[:,2]       +=  self.x_plus

        transed_bbox[:,1]       *=  self.y_scale
        transed_bbox[:,3]       *=  self.y_scale

        # Since mask is flattened
        # mask_idx        = sum([i.shape[0] for i in self.labels[:idx]])  + 1
        num_obj         = label.shape[0]  
        mask_idx        = self.mask_idx[idx]
        mask            = torch.from_numpy(self.masks[mask_idx:mask_idx + num_obj].astype(float))    

        if self.transform:
          transed_img           = self.transform(transed_img)

        if self.mask_transform:
          transed_mask          = self.mask_transform(mask)

        assert transed_img.shape == (3,800,1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]

        
        return  idx,transed_img, label, transed_mask, transed_bbox 
    
    def __len__(self):
        return len(self.images)
'''
Dataloader class
'''
class rcnn_datamodule(LightningDataModule):
    def __init__(self, dataset, batch_size=64,holdout_size = 0.2):
        super().__init__()

        self.dataset      = dataset
        self.batch_size   = batch_size
        self.holdout_size = holdout_size
        # return

    def setup(self, stage=None):
        val_split = int(self.holdout_size * len(self.dataset))  # Val set
        self.train_data, self.valid_data = random_split(self.dataset, [len(self.dataset)-val_split, val_split])
        # train_size = 100
        # val_size = 100
        # self.valid_data = Subset(self.dataset, list(range(0, train_size)))
        # self.train_data = Subset(self.dataset, list(range(train_size, train_size + val_size)))
        return
        
    def collate_fn(self, batch):
      idx, images, labels, masks, bounding_boxes = list(zip(*batch))
      return idx ,torch.stack(images), labels, masks, bounding_boxes
      # return torch.stack(images), labels, bounding_boxes, masks, idx

    def train_dataloader(self):
        return DataLoader(self.train_data, collate_fn = self.collate_fn, batch_size = self.batch_size,shuffle=True,num_workers=4)
  
    def val_dataloader(self):
        return DataLoader(self.valid_data,collate_fn = self.collate_fn, batch_size = self.batch_size,shuffle=False,num_workers=4)
