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
RPN
'''
class RPNHead(pl.LightningModule):
    # The input of the initialization of the RPN is:
    # Input:
    #       computed_anchors: the anchors computed in the dataset
    #       num_anchors: the number of anchors that are assigned to each grid cell
    #       in_channels: number of channels of the feature maps that are outputed from the backbone
    #       device: the device that we will run the model
    def __init__(self, num_anchors=3, in_channels=256, device='cuda',
                 anchors_param=dict(ratio=[[1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2]],
                                    scale=[32, 64, 128, 256, 512],
                                    grid_size=[(200, 272), (100, 136), (50, 68), (25, 34), (13, 17)],
                                    stride=[4, 8, 16, 32, 64])
                 ):
      super(RPNHead,self).__init__()

      ######################################
      # initialize RPN
      #######################################
      self.rpn_backbone = Resnet50Backbone()
      self.anchors      = self.create_anchors(anchors_param['ratio'],anchors_param['scale'],
                                          anchors_param['grid_size'],anchors_param['stride'])
      
      self.inter_layer  = nn.Sequential(
                                nn.Conv2d(256,256,kernel_size=3,padding='same'),
                                nn.BatchNorm2d(256),
                                nn.ReLU()
                                        )
      self.clas_head    = nn.Sequential(
                                nn.Conv2d(256,3,kernel_size=1,padding='same'),
                                nn.Sigmoid()
                                        )
      self.reg_head     = nn.Conv2d(256,12,kernel_size=1,padding='same')
      self.anchors_param  = anchors_param
      # Losses
      self.clas_loss    = nn.BCELoss(reduction ='sum')
      self.reg_loss     = nn.SmoothL1Loss(reduction='sum')
      # self.device       = device
      self.ground_dict  = {}

      self.train_loss       = []
      self.train_class      = []
      self.train_reg        = []
      self.val_loss         = []
      self.val_class        = []
      self.val_reg          = []
      self.save_path             =  SAVE_PATH+'RPN/'

      # pass
    # Forward each level of the FPN output through the intermediate layer and the RPN heads
    # Input:
    #       X: list:len(FPN){(bz,256,grid_size[0],grid_size[1])}
    # Ouput:
    #       logits: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       bbox_regs: list:len(FPN){(bz,4*num_anchors, grid_size[0],grid_size[1])}
    def forward(self, X):
        # inter_otp       = list(map(self.intermediate_layer,X))
        # logits          = list(map(self.classifier_head,inter_otp))
        # bbox_regs       = list(map(self.regressor_head,inter_otp))
        inter_otp       = list(map(self.inter_layer,X))
        logits          = list(map(self.clas_head,inter_otp))
        bbox_regs       = list(map(self.reg_head,inter_otp))
        return logits, bbox_regs
    
    
    # This function creates the anchor boxes for all FPN level
    # Input:
    #       aspect_ratio: list:len(FPN){list:len(number_of_aspect_ratios)}
    #       scale:        list:len(FPN)
    #       grid_size:    list:len(FPN){tuple:len(2)}
    #       stride:        list:len(FPN)
    # Output:
    #       anchors_list: list:len(FPN){(grid_size[0]*grid_size[1]*num_anchors,4)}
    def create_anchors(self, aspect_ratio, scale, grid_size, stride):
        anchors_list  = []
        # Iterate over FPNS
        for ratio,scl,grd_sz,st in zip(aspect_ratio,scale,grid_size,stride):
          #Get anchor for that particular FPN from create_anchors_single 
          anchor_fpn  = self.create_anchors_single(ratio,scl,grd_sz,st)
          anchors_list.append(anchor_fpn)

        return anchors_list



    # This function creates the anchor boxes for one FPN level
    # Input:
    #      aspect_ratio: list:len(number_of_aspect_ratios)
    #      scale: scalar
    #      grid_size: tuple:len(2)
    #      stride: scalar
    # Output:
    #       anchors: (grid_size[0]*grid_size[1]*num_acnhors,4)
    def create_anchors_single(self, aspect_ratio, scale, grid_sizes, stride):
        anchors = []
        # Iterate over aspect ratios
        for ratio in aspect_ratio:
          # Create tensors for one aspect ratio
          h                     =   scale/(np.sqrt(ratio) + 1e-6)
          w                     =   ratio*h
          y_center,x_center     =   torch.meshgrid(torch.arange(grid_sizes[0]),torch.arange(grid_sizes[1]),indexing='ij')
          x_center              =   x_center*stride
          y_center              =   y_center*stride

          # Get the centers
          x_center              +=  stride//2
          y_center              +=  stride//2

          # x1,x2,y1,y2
          x1                    =   x_center  - w/2
          x2                    =   x_center  + w/2
          y1                    =   y_center  - h/2
          y2                    =   y_center  + h/2
          anchors.append(torch.stack((x1,y1,x2,y2)).permute(1,2,0))
        # Stack all anchors and reshape them        
        anchors = torch.stack(anchors)
        anchors = anchors.view(-1,4)
        return anchors
    
    def get_anchors(self):
        return self.anchors
    
    def get_xywh_anchors(self,anch):
      '''
      anch: (n,4) tensors
      '''

      xc    = (anch[:,0] + anch[:,2])/2
      yc    = (anch[:,1] + anch[:,3])/2
      w     = anch[:,2]  - anch[:,0]
      h     = anch[:,3]  - anch[:,1]

      xc    = xc.reshape((-1,1))
      yc    = yc.reshape((-1,1))
      w     = w.reshape((-1,1))
      h     = h.reshape((-1,1))

      return torch.hstack((xc,yc,w,h))

    # This function creates the ground truth for a batch of images
    # Input:
    #      bboxes_list: list:len(bz){(number_of_boxes,4)}
    #      indexes: list:len(bz)
    #      image_shape: list:len(bz){tuple:len(2)}
    # Ouput:
    #      ground: list:len(FPN){(bz,num_anchors,grid_size[0],grid_size[1])}
    #      ground_coord: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    def create_batch_truth(self, bboxes_list, indexes, image_shape):
        ground_clas_list, ground_coord_list = MultiApply(self.create_ground_truth,
                                                # *args
                                                bboxes_list, indexes, 
                                                # **kwargs
                                                grid_sizes=self.anchors_param["grid_size"],
                                                anchors=self.anchors,
                                                image_size=image_shape)
        #  Reshape it to desired shape 
        
        ground                                = list(map(lambda x: torch.stack(x), zip(*ground_clas_list)))
        ground_coord                          = list(map(lambda x: torch.stack(x), zip(*ground_coord_list)))

        # ground                              = [torch.stack(b) for b in zip(*ground_clas_list)]
        # ground_coord                        = [torch.stack(b) for b in zip(*ground_coord_list)]
        return ground, ground_coord
    
    # This function create the ground truth for one image for all the FPN levels
    # It also caches the ground truth for the image using its index
    # Input:
    #       bboxes:      (n_boxes,4)
    #       index:       scalar (the index of the image in the total dataset)
    #       grid_size:   list:len(FPN){tuple:len(2)}
    #       anchor_list: list:len(FPN){(num_anchors*grid_size[0]*grid_size[1],4)}
    # Output:
    #       ground_clas: list:len(FPN){(num_anchors,grid_size[0],grid_size[1])}
    #       ground_coord: list:len(FPN){(4*num_anchors,grid_size[0],grid_size[1])}
    def create_ground_truth(self, bboxes, index, grid_sizes, anchors, image_size):
        key = str(index)
        if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            # groundt, ground_coord = list(map(lambda x:x.cuda(),groundt)), list(map(lambda x: x.cuda(),ground_coord))
            return groundt, ground_coord

        ground_clas     = []
        ground_coord    = []

        for grid_size_fpn, anchor_fpn in zip(grid_sizes, anchors):
          anchor_fpn                = anchor_fpn.to(self.device)
          # Initialize tensor
          fpn_clas                  = -torch.ones((anchor_fpn.shape[0],1))
          fpn_coord                 = -torch.zeros_like(anchor_fpn)
          anch_xywh                 = self.get_xywh_anchors(anchor_fpn)
          
          # Get cross bpundary masks
          crs_bnd_msk               = torch.where( ((anchor_fpn[:,0]<0) + (anchor_fpn[:,1]<0)\
                                       + (anchor_fpn[:,2]>image_size[1]) + \
                                       (anchor_fpn[:,3]>image_size[0])) >0 )[0]

          # Find IOU between anchors and bboxes
          ious                      = box_iou(anchor_fpn,bboxes)
          
          # Find positive anchors and update fpn_clas and fpn_grd
          # IOU gt is max
          anch_indx,bbox_indx     = torch.where(ious==ious.max(axis=0).values)

          if(len(anch_indx)>0):

            #Update fpn_clas
            fpn_clas[anch_indx]        = 1

            # Find the xywh anch and bbox corresponding to this 
            xywh_pos                  = anch_xywh[anch_indx]
            bbox_pos                  = bboxes[bbox_indx]

            # Calculate encoding and update ground truth
            encoding                  = self.get_encoding(xywh_pos,bbox_pos)
            fpn_coord[anch_indx]      = encoding


          # IOU greater than 0.7
          anch_indx                 = torch.where(ious.max(axis=-1).values>0.7)[0]
          bbox_indx                 = ious.argmax(axis=-1)[anch_indx]

          if(len(anch_indx)>0):

            #Update fpn_clas
            fpn_clas[anch_indx]        = 1

            # Find the xywh anch and bbox corresponding to this 
            xywh_pos                  = anch_xywh[anch_indx]
            bbox_pos                  = bboxes[bbox_indx]

            # Calculate encoding and update ground truth
            encoding                  = self.get_encoding(xywh_pos,bbox_pos)
            fpn_coord[anch_indx]      = encoding

          # Find negatives

          # IOU < 0.3
          anch_indx                 = torch.where(ious.max(axis=-1).values<0.3)[0]
          bbox_indx                 = ious.argmax(axis=-1)[anch_indx]
          if(len(anch_indx)>0):
            #Update fpn_clas
            fpn_clas[anch_indx]        = 0

          # Set cross boundary mask to -in
          fpn_clas[crs_bnd_msk]       = float('-inf')
          
          #Reshape it to required shape -- need to verify this step
          fpn_clas                    = fpn_clas.permute(1,0)
          fpn_coord                   = fpn_coord.permute(1,0)
          ground_clas.append(fpn_clas.reshape(3,grid_size_fpn[0],grid_size_fpn[1]))
          ground_coord.append(fpn_coord.reshape(3*4,grid_size_fpn[0],grid_size_fpn[1]))
        
        if len(self.ground_dict)<500:
          # self.ground_dict[key]       = (list(map(lambda x:x.cpu(),ground_clas)), list(map(lambda x:x.cpu(),ground_coord)))
          self.ground_dict[key]       = (ground_clas, ground_coord)

        return ground_clas, ground_coord

    # Compute the loss of the classifier
    # Input:
    #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
    #      n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels
    def loss_class(self, p_out, n_out):

        # torch.nn.BCELoss()
        # compute classifier's loss
        input_cls   = torch.vstack((p_out,n_out))
        target_cls  = torch.vstack((torch.ones_like(p_out), torch.zeros_like(n_out)))
        sum_count   = len(p_out) + len(n_out)
        loss        = self.clas_loss(input_cls,target_cls)
        return loss, sum_count

    # Compute the loss of the regressor
    # Input:
    #       pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
    #       pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
    def loss_reg(self, pos_target_coord, pos_out_r):
        # torch.nn.SmoothL1Loss()
        # compute regressor's loss
        loss          = self.reg_loss(pos_out_r,pos_target_coord)
        sum_count     = len(pos_target_coord)
        return loss, sum_count

    # Compute the total loss for the FPN heads
    # Input:
    #       clas_out_list: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       regr_out_list: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    #       targ_clas_list: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       targ_regr_list: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    #       l: weighting lambda between the two losses
    # Output:
    #       loss: scalar
    #       loss_c: scalar
    #       loss_r: scalar
    def compute_loss(self, clas_out_list, regr_out_list, targ_clas_list, targ_regr_list, l=10, effective_batch=150):

        # Make mask list of positive masks wrt target_clas_list -> list of positive class indices
        pos_masks           = list(map(lambda x: torch.where(x==1), targ_clas_list)) 
        neg_masks           = list(map(lambda x: torch.where(x==0), targ_clas_list)) 


        # Find total number of positive samples selected in each batch (min of batch/2 or total masks)
        pos_num             = min(effective_batch//2, sum(list(map(lambda x: len(x), pos_masks))))
        neg_num             = effective_batch - pos_num

        # Select positive vals target (num_pos,1)-> class (num_pos,12)-> reg
        # pos_clas_gt       =  torch.cat(list(map(lambda fpn_arr, mask: fpn_arr[mask], targ_clas_list , pos_masks)))[:pos_num].view(-1,1)
        pos_reg_gt        =  torch.vstack(list(map(lambda fpn_arr,mask: fpn_arr[mask[0],:,mask[1],mask[2]], targ_regr_list,pos_masks)))[:pos_num]
        
        pos_clas_pred     = torch.cat(list(map(lambda fpn_arr, mask: fpn_arr[mask], clas_out_list , pos_masks)))[:pos_num].view(-1,1)
        pos_reg_pred      =  torch.vstack(list(map(lambda fpn_arr,mask: fpn_arr[mask[0],:,mask[1],mask[2]], regr_out_list,pos_masks)))[:pos_num]
        
        # Add negative labels to it 
        # neg_clas_gt       =  torch.cat(list(map(lambda fpn_arr, mask: fpn_arr[mask], targ_clas_list , neg_masks)))[:neg_num].view(-1,1)
        neg_reg_gt        =  torch.vstack(list(map(lambda fpn_arr,mask: fpn_arr[mask[0],:,mask[1],mask[2]], targ_regr_list,neg_masks)))[:neg_num]
        
        neg_clas_pred     = torch.cat(list(map(lambda fpn_arr, mask: fpn_arr[mask], clas_out_list , neg_masks)))[:neg_num].view(-1,1)
        neg_reg_pred      = torch.vstack(list(map(lambda fpn_arr,mask: fpn_arr[mask[0],:,mask[1],mask[2]], regr_out_list,neg_masks)))[:neg_num]

        # Concatenate sample set
        # gt_clas           = torch.vstack((pos_clas_gt,neg_clas_gt))
        gt_reg            = torch.vstack((pos_reg_gt,neg_reg_gt)).view(-1,4)
        # pred_clas         = torch.vstack((pos_clas_pred,neg_clas_pred))
        pred_reg          = torch.vstack((pos_reg_pred,neg_reg_pred)).view(-1,4)


        # Call loss functions 
        loss_c,sum_c              = self.loss_class(pos_clas_pred,neg_clas_pred)
        loss_r,sum_r              = self.loss_reg(gt_reg, pred_reg)
        loss_c                    = (loss_c/(sum_c + 1e-6))
        loss_r                    = l*(loss_r/(sum_r + 1e-6))
        loss                      = loss_c  + loss_r
        # Create  
        # del pos_clas_pred, neg_clas_pred, gt_reg, pred_reg, neg_reg_pred, neg_clas_pred, neg_reg_gt, pos_reg_pred, pos_clas_pred, pos_reg_gt, pos_num, neg_num
        # gc.collect()
        return loss, loss_c, loss_r

    # Post process for the outputs for a batch of images
    # Input:
    #       out_c: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       out_r: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    #       IOU_thresh: scalar that is the IOU threshold for the NMS
    #       keep_num_preNMS: number of masks we will keep from each image before the NMS
    #       keep_num_postNMS: number of masks we will keep from each image after the NMS
    # Output:
    #       nms_clas_list: list:len(bz){(Post_NMS_boxes)} (the score of the boxes that the NMS kept)
    #       nms_prebox_list: list:len(bz){(Post_NMS_boxes,4)} (the coordinate of the boxes that the NMS kept)
    def postprocess(self, out_c, out_r, IOU_thresh=0.5, keep_num_preNMS=100, keep_num_postNMS=1000):
        out_c = list(zip(*out_c))
        out_r = list(zip(*out_r))
        nms_clas_list = []
        nms_prebox_list = []
        for c,r in zip(out_c,out_r):
          nms_clas,nms_prebox = self.postprocessImg(c,r,IOU_thresh,\
                                        keep_num_preNMS, keep_num_postNMS)
          nms_clas_list.append(nms_clas)
          nms_prebox_list.append(nms_prebox)

        return nms_clas_list, nms_prebox_list

    # Post process the output for one image
    # Input:
    #      mat_clas: list:len(FPN){(1,1*num_anchors,grid_size[0],grid_size[1])}  (score of the output boxes)
    #      mat_coord: list:len(FPN){(1,4*num_anchors,grid_size[0],grid_size[1])} (encoded coordinates of the output boxess)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def postprocessImg(self, mat_clas, mat_coord, IOU_thresh, keep_num_preNMS, keep_num_postNMS):
        anch = self.get_anchors()
        nms_clas      = []
        nms_prebox    = []
        prebox_clas   = []
        prebox_coord  = []
        for fpn_anch,fpn_clas,fpn_coord in zip(anch, mat_clas,mat_coord):
          # flatten everything to 1D or 2D
          # fpn_clas  = fpn_clas.permute(1,2,0).reshape(-1)
          # fpn_coord = fpn_coord.permute(1,2,0).reshape(-1,4)
          fpn_clas,fpn_coord  = self.flatten_rpns(fpn_clas,fpn_coord)
          xywh_anch = self.get_xywh_anchors(fpn_anch).to(device)     
          # Decode the boxes and clamp em
          decoded_box = self.decode_box(xywh_anch,fpn_coord) 
          prebox_clas.append(fpn_clas)
          prebox_coord.append(decoded_box)
        
        #Now get topK and NMS
        prebox_clas   = torch.cat(prebox_clas)
        prebox_coord  = torch.vstack(prebox_coord)
        top_k         = prebox_clas.argsort(descending=True)[:keep_num_preNMS]
        prebox_clas   = prebox_clas[top_k]
        prebox_coord  = prebox_coord[top_k]
        nms_clas,nms_prebox = self.NMS(prebox_clas,prebox_coord,keep_num_postNMS=keep_num_postNMS)

          
        return nms_clas, nms_prebox
    
    # Input:
    #       clas: (top_k_boxes) (scores of the top k boxes)
    #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def NMS(self, clas, prebox, keep_num_postNMS = 1000):
        # pass
        per_img_proposals = len(prebox)
        # Find IOU with all boxes
        ious = box_iou(prebox,prebox)
        # Get only upper diagonal
        ious_up = ious.triu(diagonal=1)
        ious_cmax = ious_up.max(0).values

        # expand to req dim -- this is to vectorize 
        ious_cmax = ious_cmax.expand(per_img_proposals, per_img_proposals).T
        
        # Get decay 
        decay = (1 - ious_up)/(1 - ious_cmax)
        decay = decay.min(dim=0).values
        decay = decay*clas
        idxs  = decay.argsort(descending=True)[:keep_num_postNMS]
        return clas[idxs], prebox[idxs]
        # return nms_clas, nms_prebox
     
    def get_encoding(self,xywh_anch, bbox):
      '''
      Both xywh_anchors and bbox should be same shape
      '''
      xc    = xywh_anch[:,0]
      yc    = xywh_anch[:,1]
      wc    = xywh_anch[:,2]
      hc    = xywh_anch[:,3]

      bbox_center = self.get_xywh_anchors(bbox)

      xgt   = bbox_center[:,0]
      ygt   = bbox_center[:,1]
      wgt   = bbox_center[:,2]
      hgt   = bbox_center[:,3]

      # Encode ground_coord 
      x_enc           = (xgt- xc)/wc
      y_enc           = (ygt -yc)/hc
      w_enc           = torch.log(wgt/(wc + 1e-6))
      h_enc           = torch.log(hgt/(hc + 1e-6))

      #Reshape for shape issue
      x_enc           = x_enc.reshape((-1,1))
      y_enc           = y_enc.reshape((-1,1))
      w_enc           = w_enc.reshape((-1,1))
      h_enc           = h_enc.reshape((-1,1))

      return torch.hstack((x_enc, y_enc, w_enc, h_enc))
    
    def flatten_rpns(self, rpn_cls, rpn_coord):
        
      # Flatten fpn_cls and fpn_coord
      rpn_cls   = rpn_cls.reshape(-1)
      rpn_coord = rpn_coord.reshape((-1,4))

      return rpn_cls, rpn_coord
        
    # def decode_one_box(self, fpn_coord, xywh_anch):
    #   xc    = xywh_anch[:,0]
    #   yc    = xywh_anch[:,1]
    #   wc    = xywh_anch[:,2]
    #   hc    = xywh_anch[:,3]
    def decode_box(self, xywh_anch, targets,img_size=(800,1088)):
      # bbox in shape x_c, y_c, w, h
      # proposals:(per_image_proposals,4)}
      # targets: (per_image_proposals, 4)

      xc              = xywh_anch[:,0]
      yc              = xywh_anch[:,1]
      wc              = xywh_anch[:,2]
      hc              = xywh_anch[:,3]

      x_center_target = targets[:,0]*wc + xc
      y_center_target = targets[:,1]*hc + yc
      w_center_target = torch.exp(targets[:,2])*wc
      h_center_target = torch.exp(targets[:,3])*hc

      x1              = torch.maximum(torch.tensor([0]).to(device)          ,x_center_target - w_center_target/2).reshape(-1,1)
      y1              = torch.maximum(torch.tensor([0]).to(device)          ,y_center_target - h_center_target/2).reshape(-1,1)
      x2              = torch.minimum(torch.tensor(img_size[1]-1).to(device),x_center_target + w_center_target/2).reshape(-1,1)
      y2              = torch.minimum(torch.tensor(img_size[0]-1).to(device),y_center_target + h_center_target/2).reshape(-1,1)

      return torch.hstack([x1,y1,x2,y2])

    def decode_one_box(self, xywh_anch, targets):
      # bbox in shape x_c, y_c, w, h
      # proposals:(per_image_proposals,4)}
      # targets: (per_image_proposals, 4)

      xc              = xywh_anch[:,0]
      yc              = xywh_anch[:,1]
      wc              = xywh_anch[:,2]
      hc              = xywh_anch[:,3]

      x_center_target = targets[0]*wc + xc
      y_center_target = targets[1]*hc + yc
      w_center_target = torch.exp(targets[2])*wc
      h_center_target = torch.exp(targets[3])*hc

      return torch.hstack([x_center_target,y_center_target,w_center_target,h_center_target])

    def decode_anch(self, gt_cls, gt_coord):
      # gt_cls: list:len(FPN){(num_anchors,grid_size[0],grid_size[1])}
      # gt_coord: list:len(FPN){(4*num_anchors,grid_size[0],grid_size[1])}

      anch          = self.get_anchors()
      decoded_anch = []
      # Iterate over FPNS - 5
      for fpn_anch, fpn_cls, fpn_coord in zip(anch,gt_cls,gt_coord):
        # Get xywh_anch
        xywh_anch = self.get_xywh_anchors(fpn_anch)

        # Flatten fpn_cls and fpn_coord
        fpn_cls, fpn_coord  = self.flatten_rpns(fpn_cls, fpn_coord)
        
        # Find where cls = 1
        true_idxs           = torch.where(fpn_cls==1)[0]
        if(len(true_idxs) > 0):
          encoded_box         = fpn_coord[true_idxs]
          anch_box            = fpn_anch[true_idxs]
          # Decode these boxes
          decoded_anch.append(anch_box)
      
      return torch.vstack(decoded_anch)
    
    def load_checkpoint(self):
      path = 'change this to where the model checkpoint is saved'
      return self.load_from_checkpoint(path)

    ##########################PL Lightning TRAINING###################3
    def training_step(self, batch, batch_idx):
      indexes,images,lbls_smp,msks_smp,boxes         = batch

      with torch.no_grad():
        backout                                     = self.rpn_backbone(images)
        fpn_feat_list                               = list(backout.values())

      gt_clas,ground_coord                           = self.create_batch_truth(boxes,indexes,images.shape[-2:])
      gt_clas_pred, gt_coord_pred                    = self(fpn_feat_list)
      train_loss, train_loss_c, train_loss_r         = self.compute_loss(gt_clas_pred, gt_coord_pred, gt_clas, ground_coord)

      self.log("train class_loss",train_loss_c,on_epoch = True)
      self.log("train reg_loss",train_loss_r,on_epoch = True)
      self.log("train_loss", train_loss,on_epoch=True, prog_bar=True)

      return {'loss':train_loss, "class_loss": train_loss_c, "reg_loss":train_loss_r}


    def validation_step(self, batch, batch_idx):
      with torch.no_grad():
        vl_indexes,vl_images,vl_lbls_smp,vl_msks_smp,vl_boxes    = batch
        
        backout                                                  = self.rpn_backbone(vl_images)
        fpn_feat_list                                            = list(backout.values())

        vl_gt_clas,vl_ground_coord                               = self.create_batch_truth(vl_boxes,vl_indexes,vl_images.shape[-2:])
        vl_gt_clas_pred, vl_gt_coord_pred                        = self(fpn_feat_list)
        val_loss, val_loss_c, val_loss_r                         = self.compute_loss(vl_gt_clas_pred, vl_gt_coord_pred, vl_gt_clas, vl_ground_coord)

        self.log("val class_loss",val_loss_c,on_epoch = True)
        self.log("val reg_loss",val_loss_r,on_epoch = True)
        self.log("val_loss", val_loss,on_epoch=True, prog_bar=True)

        return {'val_loss':val_loss, "val_class_loss": val_loss_c, "val_reg_loss":val_loss_r}

    def configure_optimizers(self):
      optim =  torch.optim.SGD(self.parameters(),lr=0.01, momentum=0.9, weight_decay=0.0001)
      lr_scheduler  = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[27,33], gamma=0.1)
      return {"optimizer": optim, "lr_scheduler": lr_scheduler}


    def training_epoch_end(self, outputs):
      '''
      Do we plot the mean loss?
      '''
      avg_train_loss        = torch.stack([x['loss'] for x in outputs]).mean().item()
      avg_train_class_loss  = torch.stack([x['class_loss'] for x in outputs]).mean().item()
      avg_train_reg_loss    = torch.stack([x['reg_loss'] for x in outputs]).mean().item()

      self.train_loss.append(avg_train_loss)
      self.train_class.append(avg_train_class_loss)
      self.train_reg.append(avg_train_reg_loss)

      print('Train Loss',self.train_loss)
      print('Train Class Loss',self.train_class)
      print('Train Reg Loss',self.train_reg)

      torch.save(self.ground_dict,self.save_path+'ground_dict.pth')

      self.log("class_loss",avg_train_class_loss,on_epoch = True)
      self.log("reg_loss",avg_train_reg_loss,on_epoch = True)
      self.log("train_loss", avg_train_loss,on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, validation_step_outputs):
      avg_val_loss        = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean().item()
      avg_val_class_loss  = torch.stack([x['val_class_loss'] for x in validation_step_outputs]).mean().item()
      avg_val_reg_loss    = torch.stack([x['val_reg_loss'] for x in validation_step_outputs]).mean().item()
      
      self.val_loss.append(avg_val_loss)
      self.val_class.append(avg_val_class_loss)
      self.val_reg.append(avg_val_reg_loss)

      print('Val Loss',self.val_loss)
      print('Val Class Loss',self.val_class)
      print('Val Reg Loss',self.val_reg)

      self.log("val_loss", avg_val_loss,on_epoch=True, prog_bar=True)
      self.log("val_class_loss",avg_val_class_loss,on_epoch = True)
      self.log("val_reg_loss",avg_val_reg_loss,on_epoch = True)
