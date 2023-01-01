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


class BoxHead(pl.LightningModule):

    def __init__(self,Classes=3,P=7 ):
        super(BoxHead,self).__init__()
        self.C=Classes
        self.P=P
        # self.pretrained_network_path  = 'checkpoint680.pth'
        # self.backbone, self.rpn = pretrained_models_680(self.pretrained_network_path)

        # initialize BoxHead
        self.inter_layer        = nn.Sequential(
                                              nn.Linear(256*self.P*self.P,1024),nn.ReLU(),nn.Linear(1024,1024),nn.ReLU()
                                            )
        
        # Classifier head
        self.classifier_head  = nn.Sequential(
                                                nn.Linear(1024,self.C+1)
                                              )
        
        # Regressor head
        self.regressor_head   = nn.Sequential(
                                                nn.Linear(1024,4*self.C)
                                              )

        # For plotting
        self.train_loss       = []
        self.train_class      = []
        self.train_reg        = []
        self.val_loss         = []
        self.val_class        = []
        self.val_reg          = []

        self.ce_loss       = nn.CrossEntropyLoss()
        self.l1_loss       = nn.SmoothL1Loss(reduction ='sum')
        self.feature_sizes = np.array([ [200, 272], [100, 136], [50, 68], [25, 34], [13, 17]])

    def create_ground_truth(self,proposals,gt_labels,bbox):
    #  This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
    #  Input:
    #       proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #       gt_labels: list:len(bz) {(n_obj)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #  Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
    #       labels: (total_proposals,1) (the class that the proposal is assigned)
    #       regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)
      # print('proposals',proposals.shape())
      # print('gt',gt_labels.shape())
      # print('bbox',bbox.shape())

      batch_size           = len(proposals)
      num_proposals        = len(proposals[0])
      total_proposals      = batch_size*num_proposals
      labels               = torch.zeros((total_proposals,1)).to(device)
      regressor_target     = torch.zeros((total_proposals,4)).to(device)
      
      idx  = -1
      for batch_idx in range(batch_size):
        # print("batch_idx",batch_idx)
        curr_proposal  = proposals[batch_idx]
        curr_gt        = gt_labels[batch_idx].to(device)
        curr_bbox      = bbox[batch_idx].to(device)
        for proposal_idx in range(len(curr_proposal)):
            proposal = curr_proposal[proposal_idx]   # should be in [x1,y1,x2,y2] format
            # find IOUs with bbox and proposals
            idx += 1
            x1_gt    = proposal[0]
            y1_gt    = proposal[1]
            x2_gt    = proposal[2]
            y2_gt    = proposal[3]
          
            # Calculate IOU 

            x1_int        = torch.maximum(x1_gt,curr_bbox[:,0])
            x2_int        = torch.minimum(x2_gt,curr_bbox[:,2])
            y1_int        = torch.maximum(y1_gt,curr_bbox[:,1])
            y2_int        = torch.minimum(y2_gt,curr_bbox[:,3])          
      
            intersection = torch.maximum(torch.tensor(0), x2_int - x1_int) * torch.maximum(torch.tensor(0), y2_int - y1_int)        
            union = (x2_gt - x1_gt) * (y2_gt - y1_gt) + (curr_bbox[:,2] - curr_bbox[:,0]) * (curr_bbox[:,3] - curr_bbox[:,1]) - intersection + 1e-6
            IOU       =    intersection/union

            if IOU.max()>0.5:
              max_idx       = torch.where(IOU == IOU.max())
              labels[idx]   = curr_gt[max_idx]
              target_bbox   = curr_bbox[max_idx][0]

              x_c_prop      = (x1_gt + x2_gt)/2
              y_c_prop      = (y1_gt + y2_gt)/2
              w_prop        = (x2_gt - x1_gt) + 1e-6
              h_prop        = (y2_gt - y1_gt) + 1e-6

              x_c_bbox      = (target_bbox[0] + target_bbox[2])/2
              y_c_bbox      = (target_bbox[1] + target_bbox[3])/2
              w_bbox        = (target_bbox[2] - target_bbox[0])
              h_bbox        = (target_bbox[3] - target_bbox[1])

              t_x           = (x_c_bbox - x_c_prop)/w_prop
              t_y           = (y_c_bbox - y_c_prop)/h_prop 
              t_w           = torch.log(w_bbox/(w_prop+1e-6))
              t_h           = torch.log(h_bbox/(h_prop+1e-6))

              # print(torch.stack((t_x,t_y,t_w,t_h)))
              regressor_target[idx]   = torch.stack((t_x,t_y,t_w,t_h))

      return labels,regressor_target

    def MultiScaleRoiAlign(self, fpn_feat_list,proposals,P=7):
    # This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
    # a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
    # Input:
    #      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
    #      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #      P: scalar
    # Output:
    #      feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
    #####################################
    # Here you can use torchvision.ops.RoIAlign check the docs
    #####################################
        feature_vectors     =     []
        for i,batch in enumerate(proposals):
          for box in batch:
            x1,y1,x2,y2   = box
            # Getting k
            w             = x2  - x1
            h             = y2  - y1
            k             = int(torch.clamp(4 + torch.log2(torch.sqrt(w*h)/224),2,5))

            # Finding dimensions of required feature pyramid
            chosen_feat   = fpn_feat_list[k-2][i]
            fpn_shape     = chosen_feat.shape[-2:]

            # Find appropriate box indexes in the chosen fpn's shape -- last fpn out can't div 800
            x_div         = 1088/fpn_shape[1]
            y_div         = 800/fpn_shape[0]
            
            box_idxs      = torch.tensor([[x1/x_div,y1/y_div,x2/x_div,y2/y_div]]).to(self.device)
            #roi_align
            roi_align     = torchvision.ops.roi_align(chosen_feat.unsqueeze(0)\
                                    ,[box_idxs],output_size=P)
            
            # List of tensors, each of shape (256*P*P)
            feature_vectors.append(roi_align.flatten())

        feature_vectors = torch.stack(feature_vectors)
      
        return feature_vectors

    def postprocess_detections(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=800, keep_num_postNMS=100,mask=False):
        # This function does the post processing for the results of the Box Head for a batch of images
        # Use the proposals to distinguish the outputs from each image
        # Input:
        #       class_logits: (total_proposals,(C+1))
        #       box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
        #       proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
        #       conf_thresh: scalar
        #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
        #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
        # Output:
        #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
        #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
        #       labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
        # We return labels in {0,1,2}
        with torch.no_grad():
          boxes               = []
          scores              = []
          labels              = []

          pre_boxes           = []
          pre_scores          = []
          pre_labels          = []

          bz                  = len(proposals)

          #  Reshape shapes to (bz,per_img_proposal,(C+1)) and (bz,per_img_proposal,4*C)
          class_logits_bz     = class_logits.view((bz,-1,4))
          box_reg_bz          = box_regression.view((bz,-1,box_regression.shape[-1]))

          #  loop over each batch, calculate outputs
          for batch in range(bz):

            # These all should be (per_img_proposal,whatever given in inp)
            class_logits_img  = class_logits_bz[batch]
            box_reg_img       = box_reg_bz[batch].view((box_reg_bz.shape[1],-1,4))
            proposals_img     = proposals[batch]

            # Find class of the boxes, and get the appropriate bbox in box_reg_img
            # Since box_reg_img is of shape (num_proposals,3,4) 
            # Remove the class 0 things, as we don't care
            conf,clas          =  class_logits_img.max(axis=-1)
            class_logits_img   =  class_logits_img[clas!=0]
            box_reg_img        =  box_reg_img[clas!=0]
            proposals_img      =  proposals_img[clas!=0]
            # Now box_reg_img should just be (non_zero_proposals,4)
            conf,clas          =  class_logits_img.max(axis=-1)
            box_reg_img        =  box_reg_img[np.arange(len(box_reg_img)),clas-1]

            #  decode box
            box_xywh_img      = self.decode_target2(proposals_img,box_reg_img)
            
            #  clamp cross boundary boxes -- now output in (per_img_proposals,4) -- [x1,y1,x2,y2] (clamped according to img size)
            box_clamped_img   = self.cvt_xywh_to_xxyy_clamped(box_xywh_img)

            #  remove classes if confidence for all no-background is less than 0.5
            high_conf_idxs     = (class_logits_img[:,1:].max(axis=-1)[0]) > conf_thresh
            
            # These should be (high_conf_nums,whatever in inp)
            class_logits_img   =  class_logits_img[high_conf_idxs]
            box_reg_img        =  box_reg_img[high_conf_idxs]
            proposals_img      =  proposals_img[high_conf_idxs]
            box_clamped_img    =  box_clamped_img[high_conf_idxs]

            # print("After high confidence sup",class_logits_img.shape)

            # Keep top K boxes (pre_NMS)
            # 1. find the confidence and class of images
            conf,clas          =  class_logits_img.max(axis=-1)
            # 2. Sort these in decending order of confidence and choose top k
            sorted_conf,sorted_idx  = conf.sort(descending = True)
            topk_idx_preNMS         = sorted_idx[:keep_num_preNMS]
            # 3. Just choose the top k boxes, now that we have indices, 
            #    these all below should just be (keep_num_preNMS, whatever inp)
            class_logits_img   =  class_logits_img[topk_idx_preNMS]
            box_reg_img        =  box_reg_img[topk_idx_preNMS]
            proposals_img      =  proposals_img[topk_idx_preNMS]
            box_clamped_img    =  box_clamped_img[topk_idx_preNMS]
            # print("After pre NMS",class_logits_img.shape)

            clas_score,clas_lbl         =  class_logits_img.max(axis=-1)
            clas_lbl                    =  clas_lbl - 1
            pre_boxes.append(box_clamped_img)
            pre_scores.append(clas_score)
            pre_labels.append(clas_lbl)

            # Apply NMS 
            chosen_bbox        =  []
            labels_bbox        =  []
            score_bbox         =  []
            # 1. loop over classes
            for clas in range(1,4):
              # 2. find indexes belonging to a particular class
              idx_clas    = (class_logits_img.max(axis=-1)[1] == clas)
              logits_clas = class_logits_img[idx_clas]
              boxes_clas  = box_clamped_img[idx_clas]
              # 3. Apply NMS for these class boxes
              conf_clas,_ = logits_clas.max(axis=-1)
              if(len(boxes_clas)==0):
                nms_bbox,nms_conf  = torch.tensor([]),torch.tensor([])
              else:
                nms_bbox, nms_conf = self.matrix_nms(boxes_clas,conf_clas,post_nms=keep_num_postNMS)
              nms_label = (clas-1)*torch.ones(len(nms_bbox))
              chosen_bbox.append(nms_bbox)
              labels_bbox.append(nms_label)
              score_bbox.append(nms_conf)
            
            # Now NMS is done, stack the images and append to output
            # img_bbox            = torch.vstack(chosen_bbox)
            # img_labels          = torch.stack(labels_bbox)
            # img_scores          = torch.stack(score_bbox)

            #Append answers
            boxes.append(chosen_bbox)
            scores.append(score_bbox)
            labels.append(labels_bbox)
            
          return boxes, scores, labels, pre_boxes, pre_scores, pre_labels

    
    
    def postprocess_detections2(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=20, keep_num_postNMS=10,mask=False):
        # This function does the post processing for the results of the Box Head for a batch of images
        # Use the proposals to distinguish the outputs from each image
        # Input:
        #       class_logits: (total_proposals,(C+1))
        #       box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
        #       proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
        #       conf_thresh: scalar
        #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
        #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
        # Output:
        #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
        #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
        #       labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
        # We return labels in {0,1,2}
        with torch.no_grad():

          boxes               = []
          scores              = []
          labels              = []
          prop                = []

          pre_boxes           = []
          pre_scores          = []
          pre_labels          = []

          bz                  = len(proposals)

          #  Reshape shapes to (bz,per_img_proposal,(C+1)) and (bz,per_img_proposal,4*C)
          class_logits_bz     = class_logits.view((bz,-1,4))
          box_reg_bz          = box_regression.view((bz,-1,box_regression.shape[-1]))

          #  loop over each batch, calculate outputs
          for batch in range(bz):

            # These all should be (per_img_proposal,whatever given in inp)
            class_logits_img  = class_logits_bz[batch]
            box_reg_img       = box_reg_bz[batch].view((box_reg_bz.shape[1],-1,4))
            proposals_img     = proposals[batch]

            if(not mask):
              # Find class of the boxes, and get the appropriate bbox in box_reg_img
              # Since box_reg_img is of shape (num_proposals,3,4) 
              # Remove the class 0 things, as we don't care
              conf,clas          =  class_logits_img.max(axis=-1)
              class_logits_img   =  class_logits_img[clas!=0]
              box_reg_img        =  box_reg_img[clas!=0]
              proposals_img      =  proposals_img[clas!=0]

            # Now box_reg_img should just be (non_zero_proposals,4)
            conf,clas          =  class_logits_img.max(axis=-1)
            box_reg_img        =  box_reg_img[np.arange(len(box_reg_img)),clas-1]
            #  decode box
            box_xywh_img      = self.decode_target2(proposals_img,box_reg_img)
            
            if(not mask):
              #  clamp cross boundary boxes -- now output in (per_img_proposals,4) -- [x1,y1,x2,y2] (clamped according to img size)
              box_clamped_img   = self.cvt_xywh_to_xxyy_clamped(box_xywh_img)
            else:
              #  Remove cross boundary boxes
              box_clamped_img,msk   = self.cvt_xywh_to_xxyy_removed(box_xywh_img)
              #have to set class_logits as well
              class_logits_img      = class_logits_img[msk]
              box_reg_img           = box_reg_img[msk]
              proposals_img         = proposals_img[msk]

            #  remove classes if confidence for all no-background is less than 0.5
            high_conf_idxs     = (class_logits_img[:,1:].max(axis=-1)[0]) > conf_thresh
            
            # These should be (high_conf_nums,whatever in inp)
            class_logits_img   =  class_logits_img[high_conf_idxs]
            box_reg_img        =  box_reg_img[high_conf_idxs]
            proposals_img      =  proposals_img[high_conf_idxs]
            box_clamped_img    =  box_clamped_img[high_conf_idxs]

            # print("After high confidence sup",class_logits_img.shape)

            # Keep top K boxes (pre_NMS)
            # 1. find the confidence and class of images
            conf,clas          =  class_logits_img.max(axis=-1)
            # 2. Sort these in decending order of confidence and choose top k
            sorted_conf,sorted_idx  = conf.sort(descending = True)
            topk_idx_preNMS         = sorted_idx[:keep_num_preNMS]
            # 3. Just choose the top k boxes, now that we have indices, 
            #    these all below should just be (keep_num_preNMS, whatever inp)
            class_logits_img   =  class_logits_img[topk_idx_preNMS]
            box_reg_img        =  box_reg_img[topk_idx_preNMS]
            proposals_img      =  proposals_img[topk_idx_preNMS]
            box_clamped_img    =  box_clamped_img[topk_idx_preNMS]
            # print("After pre NMS",class_logits_img.shape)

            clas_score,clas_lbl         =  class_logits_img.max(axis=-1)
            clas_lbl                    =  clas_lbl - 1
            pre_boxes.append(box_clamped_img)
            pre_scores.append(clas_score)
            pre_labels.append(clas_lbl)

            # Apply NMS 
            chosen_bbox        =  []
            labels_bbox        =  []
            score_bbox         =  []
            prop_bbox          =  []  
            # 1. loop over classes
            for clas in range(1,4):
              # 2. find indexes belonging to a particular class
              idx_clas    = (class_logits_img.max(axis=-1)[1] == clas)
              logits_clas = class_logits_img[idx_clas]
              boxes_clas  = box_clamped_img[idx_clas]
              proposal_clas = proposals_img[idx_clas]
              # 3. Apply NMS for these class boxes
              conf_clas,_ = logits_clas.max(axis=-1)
              if(len(boxes_clas)==0):
                continue
              nms_bbox, nms_conf,nms_proposals  = self.matrix_nms(boxes_clas,conf_clas)
              # nms_bbox, nms_conf,nms_proposals  = self.NMS(boxes_clas,conf_clas,proposal_clas,num_post_NMS=keep_num_postNMS)
              chosen_bbox.append(nms_bbox)
              labels_bbox.append((clas-1)*torch.ones(len(nms_bbox)))
              score_bbox.append(nms_conf)
              prop_bbox.append(nms_proposals)
            
            # Now NMS is done, stack the images and append to output
            # img_bbox            = torch.vstack(chosen_bbox)
            # img_labels          = torch.stack(labels_bbox)
            # img_scores          = torch.stack(score_bbox)

            #Append answers
            boxes.append(chosen_bbox)
            scores.append(score_bbox)
            labels.append(labels_bbox)
            prop.append(prop_bbox)

          return boxes, scores, labels, pre_boxes, pre_scores, pre_labels, prop

    def compute_loss(self,class_logits, box_preds, labels, regression_targets,l=10,effective_batch=150):
        # Compute the total loss of the classifier and the regressor
        # Input:
        #      class_logits: (total_proposals,(C+1)) (as outputed from forward, not passed from softmax so we can use CrossEntropyLoss)
        #      box_preds: (total_proposals,4*C)      (as outputed from forward)
        #      labels: (total_proposals,1)
        #      regression_targets: (total_proposals,4)
        #      l: scalar (weighting of the two losses)
        #      effective_batch: scalar
        # Outpus:
        #      loss: scalar
        #      loss_class: scalar
        #      loss_regr: scalar

        num_positive    = effective_batch - effective_batch//4
        num_back        = effective_batch//4

        # find idx corresponding to positive and negative anchor classes
        pos_idx               = torch.nonzero(labels!=0)
        neg_idx               = torch.nonzero(labels==0)
        
        # count number of positive and negative classes
        total_pos_num_samples = len(pos_idx)
        total_neg_num_samples = len(neg_idx)

        # sample M/2 positive cases if total_pos_num_samples>effective_batch
        if(total_pos_num_samples>num_positive):
          #shuffle indices and anchors and select the first 50 samples
          rnd_pos_idx                       = torch.randperm(total_pos_num_samples)
          rnd_neg_idx                       = torch.randperm(total_neg_num_samples)
          chosen_rnd_pos_idx                = pos_idx[rnd_pos_idx][:num_positive]
          chosen_rnd_neg_idx                = neg_idx[rnd_neg_idx][:num_back]
        else:
          chosen_rnd_pos_idx                = pos_idx
          reqd_num_indices                  = effective_batch - len(pos_idx)
          rnd_neg_idx                       = torch.randperm(total_neg_num_samples)
          chosen_rnd_neg_idx                = neg_idx[rnd_neg_idx][:reqd_num_indices]

        chosen_rnd_pos_idx                  = chosen_rnd_pos_idx.permute(1,0)
        chosen_rnd_neg_idx                  = chosen_rnd_neg_idx.permute(1,0)

        idxs                                = torch.cat([chosen_rnd_pos_idx[0],chosen_rnd_neg_idx[0]])
        idxs                                = idxs[torch.randperm(len(idxs))]

        # Converting box_preds from (total_proposals,4*C) to (total_proposals,C,4) for easiness
        box_preds_class                     = box_preds.view(len(box_preds),-1,4)
        
        # Generating targets and inputs for loss
        labels_c                            = labels[idxs].flatten().long()        
        class_logits_c                      = class_logits[idxs]
        
        # classification loss
        loss_class                          = self.ce_loss(class_logits_c,labels_c)

        #Generating targets and inputs for regression
        ind1                                = chosen_rnd_pos_idx[0]
        ind2                                = labels[ind1].flatten().long() - 1
        box_preds_c                         = box_preds_class[ind1,ind2]
        regression_targets_c                = regression_targets[ind1]

        # regression loss
        loss_regr                           = l*self.l1_loss(box_preds_c,regression_targets_c)/(len(box_preds_c) + 1e-6)

        # print("loss_clas", loss_regr,"loss_regr",loss_class)
        loss                                =loss_class +loss_regr
        return loss, loss_class, loss_regr

    def forward(self, feature_vectors):
    # Forward the pooled feature vectors through the intermediate layer and the classifier, regressor of the box head
    # Input:
    #        feature_vectors: (total_proposals, 256*P*P)
    # Outputs:
    #        class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, notice if you want to use
    #                                               CrossEntropyLoss you should not pass the output through softmax here)
    #        box_pred:     (total_proposals,4*C)
        
        intermediate_layer  = self.inter_layer(feature_vectors)
        class_logits        = self.classifier_head(intermediate_layer)
        box_pred            = self.regressor_head(intermediate_layer)
        return class_logits, box_pred

    def decode_target(self, proposals, targets):
      # bbox in shape x_c, y_c, w, h
      # proposals:(per_image_proposals,4)}
      # targets: (per_image_proposals, 4)
  
      x_c_proposal  = (proposals[0] + proposals[2])/2
      y_c_proposal  = (proposals[1] + proposals[3])/2
      w_c_proposal  = proposals[2] - proposals[0]
      h_c_proposal  = proposals[3] - proposals[1]


      x_center_target = targets[0]*w_c_proposal + x_c_proposal
      y_center_target = targets[1]*h_c_proposal + y_c_proposal
      w_center_target = torch.exp(targets[2])*w_c_proposal
      h_center_target = torch.exp(targets[3])*h_c_proposal

      return torch.hstack([x_center_target,y_center_target,w_center_target,h_center_target])

    def decode_target2(self, proposals, targets):
      # bbox in shape x_c, y_c, w, h
      # proposals:(per_image_proposals,4)}
      # targets: (per_image_proposals, 4)
  
      x_c_proposal  = (proposals[:,0] + proposals[:,2])/2
      y_c_proposal  = (proposals[:,1] + proposals[:,3])/2
      w_c_proposal  = proposals[:,2] - proposals[:,0]
      h_c_proposal  = proposals[:,3] - proposals[:,1]


      x_center_target = targets[:,0]*w_c_proposal + x_c_proposal
      y_center_target = targets[:,1]*h_c_proposal + y_c_proposal
      w_center_target = torch.exp(targets[:,2])*w_c_proposal
      h_center_target = torch.exp(targets[:,3])*h_c_proposal

      return torch.vstack([x_center_target,y_center_target,w_center_target,h_center_target]).T

    def cvt_xywh_to_xxyy_clamped(self, batch, img_size=(800,1088)):
      '''
      batch     - (N,4)
      img_size  - (Sy,Sx)
      
      otp       - (N,4)
      '''

      xc    = batch[:,0]
      yc    = batch[:,1]
      w     = batch[:,2]
      h     = batch[:,3]

      x1    = torch.maximum(torch.tensor([0]).to(device),xc  - w/2)
      x2    = torch.minimum(torch.tensor([img_size[1]]).to(device),xc  + w/2)
      y1    = torch.maximum(torch.tensor([0]).to(device),yc  - h/2)
      y2    = torch.minimum(torch.tensor([img_size[0]]).to(device),yc  + h/2)

      return torch.stack((x1,y1,x2,y2)).T

    def cvt_xywh_to_xxyy_removed(self, batch, img_size=(800,1088)):
      '''
      batch     - (N,4)
      img_size  - (Sy,Sx)
      
      otp       - (N,4)
      '''

      xc    = batch[:,0]
      yc    = batch[:,1]
      w     = batch[:,2]
      h     = batch[:,3]

      x1    = xc  - w/2
      x2    = xc  + w/2
      y1    = yc  - h/2
      y2    = yc  + h/2
 
      # Get cross bpundary masks
      non_crs_bnd_msk    = torch.where( ((x1>=0) * (y1>=0)\
                                    * (x2<=img_size[1]) * \
                                    (y2<=img_size[0])) >0 )[0]
      img   = torch.stack((x1,y1,x2,y2)).T
      img   = img[non_crs_bnd_msk]

      return img, non_crs_bnd_msk

    def NMS(self,class_bbox,class_conf, IOU_thresh=0.5):
      '''
      class_bbox      - (num_boxes_class,4) -> [x1,y1,x2,y2]
      class_conf      - (num_boxes_class)   -> conf values of classes  

      returns tensor - (num_afterIOU, 4) and conf : (num_after_IOU)
      '''

      #1. Sort images according to conf
      conf,idxs       = class_conf.sort(descending=True)
      class_bbox      = class_bbox[idxs]
      chosen_flag     = torch.ones(idxs.shape[0])
      #2. do NMS
      for img_idx in range(len(idxs)):
        # If image not chosen don't consider
        if(chosen_flag[img_idx] == 0):
          continue
        img1                            = class_bbox[img_idx]
        x1_img1,y1_img1,x2_img1,y2_img1 = img1
        for img2_idx in range(img_idx+1,len(idxs)):
          img2                            = class_bbox[img2_idx]          
          x1_img2,y1_img2,x2_img2,y2_img2 = img2
          
          #Calculate IOU
          x1_int        = torch.maximum(x1_img1,x1_img2)
          x2_int        = torch.minimum(x2_img1,x2_img2)
          y1_int        = torch.maximum(y1_img1,y1_img2)
          y2_int        = torch.minimum(y2_img1,y2_img2)          
    
          intersection = torch.maximum(torch.tensor(0), x2_int - x1_int) * torch.maximum(torch.tensor(0), y2_int - y1_int)        
          union = (x2_img1 - x1_img1) * (y2_img1 - y1_img1) + (x2_img2 - x1_img2) * (y2_img2 - y1_img2) - intersection + 1e-6
          IOU       =    intersection/union
          
          if(IOU>IOU_thresh):
            chosen_flag[img2_idx]  = 0
      return class_bbox[chosen_flag==1], class_conf[chosen_flag==1]
    
    def matrix_nms(self,box,score,decay=0.9,post_nms=10):
      # box - tensor (per_img_proposals,4) - ([x1,x2,y1,y2])
      # score - tensor(per_img_proposals,1) - (scores for these boxes)
      # decay - decay factor (float) -> default 0.9
      #-----------------
      # Output - idxs -- returns the indices of selected items 
      
      per_img_proposals = len(box)
      # Find IOU with all boxes
      ious = box_iou(box,box)
      # Get only upper diagonal
      ious_up = ious.triu(diagonal=1)
      ious_cmax = ious_up.max(0).values

      # expand to req dim -- this is to vectorize 
      ious_cmax = ious_cmax.expand(per_img_proposals, per_img_proposals).T
      
      # Get decay 
      decay = (1 - ious_up)/(1 - ious_cmax)
      decay = decay.min(dim=0).values
      decay = decay*score
      idxs  = decay.argsort(descending=True)[:post_nms]
      return box[idxs], score[idxs]

    def NMS2(self,class_bbox,class_conf, proposals, IOU_thresh=0.5,num_post_NMS=100):
      '''
      class_bbox      - (num_boxes_class,4) -> [x1,y1,x2,y2]
      class_conf      - (num_boxes_class)   -> conf values of classes  

      returns tensor - (num_afterIOU, 4) and conf : (num_after_IOU)
      '''

      #1. Sort images according to conf
      conf,idxs       = class_conf.sort(descending=True)
      class_bbox      = class_bbox[idxs]
      chosen_flag     = torch.ones(idxs.shape[0])
      #2. do NMS
      for img_idx in range(len(idxs)):
        # If image not chosen don't consider
        if(chosen_flag[img_idx] == 0):
          continue
        img1                            = class_bbox[img_idx]
        x1_img1,y1_img1,x2_img1,y2_img1 = img1
        for img2_idx in range(img_idx+1,len(idxs)):
          img2                            = class_bbox[img2_idx]          
          x1_img2,y1_img2,x2_img2,y2_img2 = img2
          
          #Calculate IOU
          x1_int        = torch.maximum(x1_img1,x1_img2)
          x2_int        = torch.minimum(x2_img1,x2_img2)
          y1_int        = torch.maximum(y1_img1,y1_img2)
          y2_int        = torch.minimum(y2_img1,y2_img2)          
    
          intersection = torch.maximum(torch.tensor(0), x2_int - x1_int) * torch.maximum(torch.tensor(0), y2_int - y1_int)        
          union = (x2_img1 - x1_img1) * (y2_img1 - y1_img1) + (x2_img2 - x1_img2) * (y2_img2 - y1_img2) - intersection + 1e-6
          IOU       =    intersection/union
          
          if(IOU>IOU_thresh):
            chosen_flag[img2_idx]  *= 0.9
      #Sort according to chosen_flag
      _,idx         = chosen_flag.sort(descending=True)
      return class_bbox[idx][:num_post_NMS], class_conf[idx][:num_post_NMS], proposals[idx][:num_post_NMS]

    def get_m_s_t(self,preds,target,sort_scores,thresh=0.5):
      # Functoin to get the matches and trues for a particular class in img

        pred_org_shape = len(preds)
        tar_org_shape = len(target)

        p_id,t_id = torch.meshgrid(torch.arange(len(preds)),torch.arange(len(target)), indexing='ij')
        p_id,t_id = p_id.ravel(),t_id.ravel()

        pred_all  = preds[p_id]
        target_all= target[t_id]
        
        w_pred    = pred_all[:,3] - pred_all[:,1] 
        h_pred    = pred_all[:,-1]  - pred_all[:,2] 
        area_pred = w_pred*h_pred

        w_target    = target_all[:,3] - target_all[:,1] 
        h_target    = target_all[:,-1]  - target_all[:,2] 
        area_target = w_target*h_target

        x1_int        = torch.maximum(pred_all[:,0] ,target_all[:,0])
        x2_int        = torch.minimum(pred_all[:,2] ,target_all[:,2])
        y1_int        = torch.maximum(pred_all[:,1],target_all[:,1])
        y2_int        = torch.minimum(pred_all[:,-1] ,target_all[:,-1])   

        area_int      = torch.maximum(torch.tensor(0), x2_int - x1_int) * torch.maximum(torch.tensor(0), y2_int - y1_int)  

        IOU           = (area_int)/(area_pred + area_target - area_int + 1e-6)
        IOU         = (IOU.reshape(pred_org_shape,tar_org_shape) )
        IOU         = (IOU>thresh)*IOU

        # To check if val present
        checker     = set()

        sort_row    = IOU.argsort(axis=1,descending=True)
        matches     = []
        scores      = []

        for i in range(pred_org_shape):
          req_row = sort_row[i]
          IOU_row = IOU[i]

          for true_idx in req_row:
            if(IOU_row[true_idx]  ==  0):
              matches.append(0)
              scores.append(sort_scores[i])
              continue
            if(true_idx not in checker):
              checker.add(true_idx)
              matches.append(1)
              scores.append(sort_scores[i])
        
        return matches,scores

    def average_precision(self,match_values,score_values,total_trues):
      max_score   = max(score_values)
      ln          = torch.linspace(0.6,max_score,100)
      precision_mat = torch.zeros(101)
      recall_mat    = torch.zeros(101)
      score_values  = torch.stack(score_values)
      match_values  = torch.tensor(match_values)
      for i,th in enumerate(ln):

        thresh_flag = score_values>th
        TP          = match_values[thresh_flag].sum()
        total_positives = len(match_values)
        precision       = 1
        if total_positives >0:
          precision = TP/total_positives

        recall          = 1
        if total_positives >0:
          recall        = TP/total_trues
        
        precision_mat[i] = precision
        recall_mat[i]     = recall
      
      recall_mat[100]     = 0
      precision_mat[100]  = 1
      sorted_ind        = torch.argsort(recall_mat)
      sorted_recall     = recall_mat[sorted_ind]
      sorted_precision  = precision_mat[sorted_ind]
      
      area              = metrics.auc(recall_mat, precision_mat)

      return area

    def mAP(self, gt_logits,gt_boxes,pred_logits, pred_box, pred_scores_all, proposals):
      '''
      gt_logits   - (total_proposals,1)
      gt_boxes    - (total_proposals,4) -(tx,ty,tw.th)

      pred_logits - list:len(bz){(post_NMS_boxes_per_image)}   
      pred_scores_all - list:len(bz){(post_NMS_boxes_per_image)}   
      pred_box    - list:len(bz){(post_NMS_boxes_per_image,4)} - (x1,y1,x2,y2)
      This function is called only after NMS 
      '''
      bz            = len(pred_logits)
      # Cvt to (bz,per_img_proposal,whatever)
      gt_logits_im  = gt_logits.view(bz,-1,1)
      gt_boxes_im   = gt_boxes.view(bz,-1,4)

      # Initialize matches, trues and scores
      matches       = {0:[],1:[],2:[]}
      trues         = {0:0,1:0,2:0}
      scores        = {0:[],1:[],2:[]}

      # For each image
      for i in range(bz):
        gt_class    = gt_logits_im[i]
        gt_bbox     = self.cvt_xywh_to_xxyy_clamped(self.decode_target2(proposals[i],gt_boxes_im[i])) 
        pred_class  = pred_logits[i]
        pred_bbox   = pred_box[i]
        pred_scores = pred_scores_all[i]

        # For each class
        for c in range(3):
          gt_cls_img_flag     =   gt_class==c
          pred_cls_img_flag   =   pred_class==c

          trues[c]            += gt_cls_img_flag.sum()

          gt_box_cls          =  gt_bbox[gt_cls_img_flag.flatten()]
          pred_box_cls        =  pred_bbox[pred_cls_img_flag.flatten()]

          sort_scores,sort_ind = pred_scores[pred_cls_img_flag.flatten()].sort(descending=True)  
          pred_box_cls        =  pred_box_cls[sort_ind]
          m,s                 =  self.get_m_s_t(pred_box_cls,gt_box_cls,sort_scores)

          matches[c].extend(m)
          scores[c].extend(s)

      ap  = 0
      cnt = 0
      mAP = 0
      for c in range(3):
        if(len(matches[c])>0):
          ap                  +=  self.average_precision(matches[c],scores[c],trues[c])
          cnt+=1
      if(cnt>0):
        mAP   = ap/cnt

      return mAP



    '''
    ###########################################################################
    ## 5.    PYTORCH LIGHTNING TRAINING LOOP 
    ###########################################################################
    '''
    def training_step(self, batch, batch_idx,keep_topK=200):
      indexes,images,lbls_smp,msks_smp,boxes         = batch

      images                                        = images.to(device)
      # Load RPNs
      backout                                        = backbone(images.float())
      im_lis                                         = ImageList(images, [(800, 1088)]*images.shape[0])
      
      rpnout                                         = rpn(im_lis, backout)
      proposals                                      = [proposal[0:keep_topK,:] for proposal in rpnout[0]]
      fpn_feat_list                                  = list(backout.values())

      # ROI Align 
      roi_align_proposals                            = self.MultiScaleRoiAlign(fpn_feat_list,proposals)

      # Create ground truth
      labels, regressor_target                       = self.create_ground_truth(proposals,lbls_smp,boxes)

      # Forward function
      clas_logits,  box_pred                         = self(roi_align_proposals.detach())
      
      # Loss
      train_loss, train_loss_c, train_loss_r         = self.compute_loss(clas_logits, box_pred, labels, regressor_target,l=10,effective_batch=150)
      return {'loss':train_loss, "class_loss": train_loss_c, "reg_loss":train_loss_r}


    def validation_step(self, batch, batch_idx,keep_topK=200):
      indexes,images,lbls_smp,msks_smp,boxes         = batch
      # Load RPNs
      backout                                        = backbone(images)
      im_lis                                         = ImageList(images, [(800, 1088)]*images.shape[0])
      rpnout                                         = rpn(im_lis, backout)
      proposals                                      = [proposal[0:keep_topK,:] for proposal in rpnout[0]]
      fpn_feat_list                                  = list(backout.values())

      # ROI Align 
      roi_align_proposals                            = self.MultiScaleRoiAlign(fpn_feat_list,proposals)

      # Create ground truth
      labels, regressor_target                       = self.create_ground_truth(proposals,lbls_smp,boxes)

      # Forward function
      clas_logits,  box_pred                         = self(roi_align_proposals.detach().to(device))

      # Loss
      val_loss, val_loss_c, val_loss_r               = self.compute_loss(clas_logits, box_pred, labels, regressor_target,l=10,effective_batch=150)

      return {'val_loss':val_loss, "val_class_loss": val_loss_c, "val_reg_loss":val_loss_r}

    def configure_optimizers(self):
      optim = torch.optim.Adam(self.parameters(), lr=0.001)
      lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [10, 15], gamma=0.1)
      return {"optimizer": optim, "lr_scheduler": lr_scheduler}


    def load_checkpoint(self):
      # -- complete this
      path = 'change this to where the model checkpoint is saved'
      return self.load_from_checkpoint(path)


    def training_epoch_end(self, outputs):

      avg_train_loss        = torch.stack([x['loss'] for x in outputs]).mean().item()
      avg_train_class_loss  = torch.stack([x['class_loss'] for x in outputs]).mean().item()
      avg_train_reg_loss    = torch.stack([x['reg_loss'] for x in outputs]).mean().item()

      self.train_loss.append(avg_train_loss)
      self.train_class.append(avg_train_class_loss)
      self.train_reg.append(avg_train_reg_loss)

      print("train_loss", self.train_loss)
      print("train_class", self.train_class)
      print("train_reg", self.train_reg)

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

      print("val_loss", self.val_loss)
      print("val_class", self.val_class)
      print("val_reg", self.val_reg)

      self.log("val_loss", avg_val_loss,on_epoch=True, prog_bar=True)
      self.log("val_class_loss",avg_val_class_loss,on_epoch = True)
      self.log("val_reg_loss",avg_val_reg_loss,on_epoch = True)
