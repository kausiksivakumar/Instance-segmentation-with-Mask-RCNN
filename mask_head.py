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


class MaskHead(pl.LightningModule):
    def __init__(self,Classes=3,P=14):
        super(MaskHead,self).__init__()
        self.C          = Classes
        self.P          = P
        self.fpn        = Resnet50Backbone()
        self.save_path  = SAVE_PATH+'Mask/'
        # initialize MaskHead
        self.mask_layer = nn.Sequential(
                                          nn.Conv2d(256,256,kernel_size=3,padding='same'),
                                          nn.ReLU(),
                                          nn.Conv2d(256,256,kernel_size=3,padding='same'),
                                          nn.ReLU(),
                                          nn.Conv2d(256,256,kernel_size=3,padding='same'),
                                          nn.ReLU(),
                                          nn.Conv2d(256,256,kernel_size=3,padding='same'),
                                          nn.ReLU(),
                                          nn.ConvTranspose2d(256,256,kernel_size=2,stride=2),
                                          nn.ReLU(),
                                          nn.Conv2d(256,3,kernel_size=1),
                                          nn.Sigmoid()
                                        )
        self.loss     =  nn.BCELoss(reduction='sum')
        self.gt_dict  = {}
        # For plotting
        self.train_loss       = []
        self.train_class      = []
        self.train_reg        = []
        self.val_loss         = []
        self.val_class        = []
        self.val_reg          = []
            
    def MultiScaleRoiAlign(self, fpn_feat_list,proposals,P=14):
    # This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
    # a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
    # Input:
    #      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
    #      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #      P: scalar
    # Output:
    #      feature_vectors: (total_proposals, 256, P, P)  (make sure the ordering of the proposals are the same as the ground truth creation)
    #####################################
    # Here you can use torchvision.ops.RoIAlign check the docs
    #####################################
        feat_bz = []
        for i,batch in enumerate(proposals):
          feature_vectors     =     []
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
            
            box_idxs      = torch.tensor([[x1/x_div,y1/y_div,x2/x_div,y2/y_div]]).to(device)
            #roi_align
            roi_align     = torchvision.ops.roi_align(chosen_feat.unsqueeze(0)\
                                    ,[box_idxs],output_size=P).squeeze()
            
            # List of tensors, each of shape (256,P,P)
            feature_vectors.append(roi_align)
          feat_bz.append(torch.stack(feature_vectors))
        # feature_vectors = torch.stack(feature_vectors)
        return feat_bz

    # This function does the pre-prossesing of the proposals created by the Box Head (during the training of the Mask Head)
    # and create the ground truth for the Mask Head
    #
    # Input:
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)  ([t_x,t_y,t_w,t_h])
    #       gt_labels: list:len(bz) {(n_obj)}
    #       proposals: len(bz) (per_img_proposals,4)        ([x1,x2,y1,y2])
    #       fpn_feat_list:  len(FPN){(bz,256,H_feat,W_feat)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       masks: list:len(bz){(n_obj,800,1088)}
    #       IOU_thresh: scalar (threshold to filter regressed with low IOU with a bounding box)
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)} ([x1,y1,x2,y2] format)
    #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}  (top category of each regressed box)
    #       gt_masks: list:len(bz){(post_NMS_boxes_per_image,2*P,2*P)}
    def preprocess_ground_truth_creation(self, idxs, class_logits, box_regression, gt_labels,proposals,bbox ,masks ,fpn_feat_list, img_size=(800,1088), decay=0.9, keep_num_preNMS=300, keep_num_postNMS=100,debug=False):
        start_all = time.time()
        with torch.no_grad():
          # Reshape proposals to 
          bz                = len(bbox)
          boxes             = []
          scores            = []
          proposals_nms     = []
          labels            = []
          gt_masks          = []

          # Similarly for box_regression and class_logits - (bz, per_img, 4)
          proposals     = torch.vstack(proposals).reshape(bz,-1,4)
          class_logits  = class_logits.reshape((bz,-1,class_logits.shape[-1]))
          
          # Box regs should be (bz,per_img_proposals,3,4), changing proposals to same
          box_regression  = box_regression.reshape((bz,-1,3,4))
          proposals       = proposals.unsqueeze(2).repeat_interleave(3,2)

          for batch in range(bz):
            key = str(idxs[batch])
            if key in self.gt_dict:
              box,score,gt_label,gt_mask  = self.gt_dict[key]

            else:
              # proposals and box_reg - (per_img,3,4); class_logits - (per_img,4)
              proposals_b       = proposals[batch]
              class_logits_b    = class_logits[batch]
              box_regression_b  = box_regression[batch] 
              bbox_b            = bbox[batch].to(device)
              mask_b            = masks[batch].to(device)
              label_b           = gt_labels[batch].to(device)
              
              # Remove cross boundary masks for all
              # 1. Decode box regressions to x1,y1,x2,y2

              box,_,nc_mask,cls = self.decode_box_reg(box_regression_b, proposals_b, img_size=img_size)
              
              # 2. Take out all cross boundary masks both should be of shape (in_bnd,4)
              box               = box[nc_mask]
              proposals_b       = proposals_b[nc_mask]

              # 3. Choose boxes that have IOU>0.5 with ground truth box -- ious (in_bnd,num_boxes)
              ious               = box_iou(box,bbox_b)
              # a. choosing indexes greater than 0.5
              max_overlap_ind    = torch.where(ious.max(axis=-1).values>0.5)
              # b. choosing num_pre_NMS boxes
              box                =  box[max_overlap_ind][:keep_num_preNMS]
              proposals_b        =  proposals_b[max_overlap_ind][:keep_num_preNMS]
              cls                =  cls[max_overlap_ind][:keep_num_preNMS]
              score              =  ious[max_overlap_ind].max(axis=-1).values[:keep_num_preNMS]

              # 4. NMS -- gets refined bbox
              post_nms_idxs      =  self.matrix_nms(box,score,cls,decay)[:keep_num_postNMS]
              box                =  box[post_nms_idxs]
              cls                =  cls[post_nms_idxs]
              score              =  score[post_nms_idxs]
              proposals_b        =  proposals_b[post_nms_idxs]

              # Mask reshaped to required shape:
              # Find object index for each box via iou
              ious               =  box_iou(box,bbox_b)

              # Now lbl - tensor(100) and  mask- tensor(100,800,1088)
              lbl                =  ious.max(axis=-1).indices #This would be in [0,1,2], not the true class label
              
              msk                =  mask_b[lbl]
              
              gt_msk             =  list(map(lambda msk,b: msk[b[1].int():b[3].int(),b[0].int():b[2].int()],msk,box))


              gt_mask             =  torch.vstack(list(map(lambda tns: \
                                    nn.functional.interpolate(tns.unsqueeze(0).unsqueeze(0).float(),size=(28,28), mode='nearest-exact').squeeze(0), gt_msk)))
              gt_label            = label_b[lbl]
              if len(self.gt_dict)<4000:
                self.gt_dict[key] = (box,score,gt_label,gt_mask)  

              if(debug):
                msk_box            =  torch.zeros_like(msk)
                # for idx,b in enumerate(box):
                #   msk_box[idx,b[1].int():b[3].int(),b[0].int():b[2].int()] = 1
                def get_box_mask(idx):
                  b       = box[idx]
                  msk_box[idx,b[1].int():b[3].int(),b[0].int():b[2].int()] = 1
                  return msk_box[idx]
                msk_box   = torch.stack(list(map(get_box_mask,torch.arange(len(box)))))
                rnd_idx = torch.randint(len(msk),size=(1,))[0]
                print("Box")
                plt.imshow(msk_box[rnd_idx].cpu().numpy())
                plt.show()
                plt.close()
                print("Mask")
                plt.imshow(msk[rnd_idx].cpu().numpy())
                plt.show()
                plt.close()
                print("Intersections")
                plt.imshow(gt_msk[rnd_idx].cpu().numpy())
                plt.show()
                plt.close()
                print("Rescaled - 28x28")
                plt.imshow(gt_mask[rnd_idx].cpu().numpy())

            boxes.append(box)
            scores.append(score)
            # proposals_nms.append(proposals_b)
            labels.append(gt_label) # Gets the true class label for all refined boxes
            gt_masks.append(gt_mask)

            del box,score,gt_label,gt_mask
            gc.collect()

          return boxes, scores, labels, gt_masks


    # This function does the pre-prossesing of the proposals created by the Box Head (during the training of the Mask Head)
    # and create the ground truth for the Mask Head
    #
    # Input:
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)  ([t_x,t_y,t_w,t_h])
    #       gt_labels: list:len(bz) {(n_obj)}
    #       proposals: len(bz) (per_img_proposals,4)        ([x1,x2,y1,y2])
    #       fpn_feat_list:  len(FPN){(bz,256,H_feat,W_feat)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       masks: list:len(bz){(n_obj,800,1088)}
    #       IOU_thresh: scalar (threshold to filter regressed with low IOU with a bounding box)
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)} ([x1,y1,x2,y2] format)
    #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}  (top category of each regressed box)
    #       gt_masks: list:len(bz){(post_NMS_boxes_per_image,2*P,2*P)}
    def preprocess_ground_truth_creation2(self, class_logits, box_regression, gt_labels,proposals,bbox ,masks ,fpn_feat_list, img_size=(800,1088), decay=0.9, keep_num_preNMS=800, keep_num_postNMS=100,debug=False):
        start_all = time.time()
        with torch.no_grad():
          # Reshape proposals to 
          bz                = len(bbox)
          boxes             = []
          scores            = []
          proposals_nms     = []
          labels            = []
          gt_masks          = []

          # Similarly for box_regression and class_logits - (bz, per_img, 4)
          proposals     = torch.vstack(proposals).reshape(bz,-1,4)
          class_logits  = class_logits.reshape((bz,-1,class_logits.shape[-1]))
          
          # Box regs should be (bz,per_img_proposals,3,4), changing proposals to same
          box_regression  = box_regression.reshape((bz,-1,3,4))
          proposals       = proposals.unsqueeze(2).repeat_interleave(3,2)

          for batch in range(bz):
            # proposals and box_reg - (per_img,3,4); class_logits - (per_img,4)
            proposals_b       = proposals[batch]
            class_logits_b    = class_logits[batch]
            box_regression_b  = box_regression[batch] 
            bbox_b            = bbox[batch].to(device)
            mask_b            = masks[batch].to(device)
            label_b           = gt_labels[batch].to(device)
            
            # Remove cross boundary masks for all
            # 1. Decode box regressions to x1,y1,x2,y2

            box,_,nc_mask,cls = self.decode_box_reg(box_regression_b, proposals_b, img_size=img_size)
            # print("--- %s 1_seconds ---" % (time.time() - start_all))
            
            # 2. Take out all cross boundary masks both should be of shape (in_bnd,4)
            box               = box[nc_mask]
            proposals_b       = proposals_b[nc_mask]
            # print("--- %s 2_seconds ---" % (time.time() - start_all))

            # 3. Choose boxes that have IOU>0.5 with ground truth box -- ious (in_bnd,num_boxes)
            ious               = box_iou(box,bbox_b)
            # a. choosing indexes greater than 0.5
            max_overlap_ind    = torch.where(ious.max(axis=-1).values>0.5)
            # b. choosing num_pre_NMS boxes
            box                =  box[max_overlap_ind][:keep_num_preNMS]
            proposals_b        =  proposals_b[max_overlap_ind][:keep_num_preNMS]
            cls                =  cls[max_overlap_ind][:keep_num_preNMS]
            score              =  ious[max_overlap_ind].max(axis=-1).values[:keep_num_preNMS]
            # print("--- %s 3_seconds ---" % (time.time() - start_all))

            # 4. NMS -- gets refined bbox
            post_nms_idxs      =  self.matrix_nms(box,score,cls,decay)[:keep_num_postNMS]
            box                =  box[post_nms_idxs]
            cls                =  cls[post_nms_idxs]
            score              =  score[post_nms_idxs]
            proposals_b        =  proposals_b[post_nms_idxs]
            # print("--- %s 4_seconds ---" % (time.time() - start_all))

            # Mask reshaped to required shape:
            # Find object index for each box via iou
            ious               =  box_iou(box,bbox_b)
            # print("--- %s 5a_seconds ---" % (time.time() - start_all))

            # Now lbl - tensor(100) and  mask- tensor(100,800,1088)
            lbl                =  ious.max(axis=-1).indices #This would be in [0,1,2], not the true class label
            # print("--- %s 5b_seconds ---" % (time.time() - start_all))
            
            msk                =  mask_b[lbl]
            # print("--- %s 5c_seconds ---" % (time.time() - start_all))
            
            # print("--- %s 5_seconds ---" % (time.time() - start_all))
            
            # Now convert boxes to pixel masks (vectorized way to get intersection)
            msk_box            =  torch.zeros_like(msk)
            
            start_loop         =  time.time()
            # for idx,b in enumerate(box):
            #   msk_box[idx,b[1].int():b[3].int(),b[0].int():b[2].int()] = 1
            def get_box_mask(idx):
              b       = box[idx]
              msk_box[idx,b[1].int():b[3].int(),b[0].int():b[2].int()] = 1
              return msk_box[idx]
            msk_box   = torch.stack(list(map(get_box_mask,torch.arange(len(box)))))
            # print("--- %s loop_seconds ---" % (time.time() - start_loop))

            #Now msk_box is raw binary masks for each bbox
            # Get intersection -- just elementwise mult
            # gt_mask            =  1*((msk_box*msk)>0)
            gt_mask            =  torch.mul(msk_box,msk)>0
            # print("--- %s 6_seconds ---" % (time.time() - start_all))

            if(debug):
              rnd_idx = torch.randint(len(msk_box),size=(1,))[0]
              print("box_pixelated")
              plt.imshow(msk_box[rnd_idx].cpu().numpy())
              plt.show()
              plt.close()
              print("mask_gt")
              plt.imshow(msk[rnd_idx].cpu().numpy())
              plt.show()
              plt.close()
              print("mask_int")
              plt.imshow(gt_mask[rnd_idx].cpu().numpy())
              plt.show()
              plt.close()

            #Reshape gt_mask to required shape (i.e) from (100, 800,1088) to (100, 28,28): nearest-exact for nearest neighbor ... which must be for mask interpolation
            gt_mask            =  nn.functional.interpolate(gt_mask.unsqueeze(0).unsqueeze(0).float(),size=(len(gt_mask),28,28), mode='nearest-exact').squeeze()
            # print("--- %s 7_seconds ---" % (time.time() - start_all))
            
            boxes.append(box)
            scores.append(score)
            proposals_nms.append(proposals_b)
            labels.append(label_b[lbl]) # Gets the true class label for all refined boxes
            gt_masks.append(gt_mask)
            # print("--- %s 8_seconds ---" % (time.time() - start_all))

            del box,score,proposals_b,lbl,gt_mask
            gc.collect()
            # print("--- %s all_seconds ---" % (time.time() - start_all))

          return boxes, scores, labels, gt_masks
    
    def matrix_nms(self,box,score,cls,decay=0.9):
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

      return decay.argsort(descending=True)


    def decode_box_reg(self,box,proposals,img_size=(800,1088)):
        # Function to decode box_head outputs -- needed to remove cross boundary masks
        # box - tensor(per_img_proposals,3,4) ([tx,ty,tw,th])
        # proposals - tensor(per_img_proposals,3,4) ([x1,y1,x2,y2])
        # Output
        # cross boundary masks      - (per_img_proposals,3)
        # box                       - (per_img_proposals,3,4) ([xc,yc,wc,hc])
        # ret_box                   - (per_img_proposals,3,4)  ([x1,y1,x2,y2])
        # class                     - (per_img_proposals,1) -{0,1,2}

        ret_box             = torch.zeros_like(box)
        x1_p,y1_p,x2_p,y2_p = proposals[:,:,0],proposals[:,:,1],proposals[:,:,2],proposals[:,:,3]
        xc_p    = (x1_p + x2_p)/2
        yc_p    = (y1_p + y2_p)/2
        wc_p    = (x2_p - x1_p)
        hc_p    = (y2_p - y1_p)

        # Decode box
        # * -> elementwise multiplication
        box[:,:,0]  = box[:,:,0] * wc_p + xc_p
        box[:,:,1]  = box[:,:,1] * hc_p + yc_p
        box[:,:,2]  = wc_p*torch.exp(box[:,:,2])
        box[:,:,3]  = hc_p*torch.exp(box[:,:,3])

        # Change it to x1,y1,x2,y2
        box_x1      = box[:,:,0] - box[:,:,2]/2
        box_y1      = box[:,:,1] - box[:,:,3]/2
        box_x2      = box[:,:,0] + box[:,:,2]/2
        box_y2      = box[:,:,1] + box[:,:,3]/2

        # Get non_cross_boundary_masks
        nc_mask     = torch.where(((box_x1>=0)*(box_y1>=0)*(box_x2<img_size[1])*(box_y2<img_size[0])) >0)
        ret_box[:,:,0]  = box_x1
        ret_box[:,:,1]  = box_y1
        ret_box[:,:,2]  = box_x2
        ret_box[:,:,3]  = box_y2

        #Return all
        return ret_box, box, nc_mask, nc_mask[1]


    # This function does the post processing for the result of the Mask Head for a batch of images. It project the predicted mask
    # back to the original image size
    # Use the regressed boxes to distinguish between the images
    # Input:
    #       masks_outputs: (total_boxes,C,2*P,2*P)
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)} ([x1,y1,x2,y2] format)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}  (top category of each regressed box)
    #       image_size: tuple:len(2)
    # Output:
    #       projected_masks: list:len(bz){(post_NMS_boxes_per_image,image_size[0],image_size[1]
    #       projected_masks: tensor:(len(bz)post_NMS_boxes_per_image,image_size[0],image_size[1]

    def postprocess_mask(self, masks_outputs, boxes, labels, image_size=(800,1088),plot=False):
        projected_masks = []
        total_boxes     = len(masks_outputs)
        bz              = len(boxes)
        # Shape total_boxes,2*P,2*P
        valid_masks     = masks_outputs[range(total_boxes),torch.cat(labels).long() - 1]
        # Reshape -- total_boxes,800,1088
        valid_masks     = nn.functional.interpolate(valid_masks.unsqueeze(0).float(),size=(800,1088), mode='bilinear').squeeze(0)
        # bz,per_img_box,800,1088        
        valid_masks     = valid_masks.reshape(bz,-1,800,1088)
        
        return valid_masks

    # Compute the total loss of the Mask Head
    # Input:
    #      mask_output: (total_boxes,C,2*P,2*P)
    #      labels: (total_boxes)
    #      gt_masks: (total_boxes,2*P,2*P)
    # Output:
    #      mask_loss
    def compute_loss(self,mask_output,labels,gt_masks):
        total_boxes = len(mask_output)
        num_pixels  = 4*(self.P**2)
        mask_lbl    = mask_output[range(total_boxes),labels.long() - 1] #Subtracting 1 to make labels to {0,1,2}
        mask_loss   = self.loss(mask_lbl,gt_masks)
        
        mask_loss   = mask_loss/num_pixels
        mask_loss   = mask_loss/total_boxes
        return mask_loss

    # Forward the pooled feature map Mask Head
    # Input:
    #        features: (total_boxes, 256,P,P)
    # Outputs:
    #        mask_outputs: (total_boxes,C,2*P,2*P)
    def forward(self, features):
        mask_outputs  = list(map(lambda x: self.mask_layer(x),features))
        return mask_outputs
    
    def load_checkpoint(self):
      path = 'change this to where the model checkpoint is saved'
      return self.load_from_checkpoint(path)

    
    ####################Forward function###################################
    def training_step(self, batch, batch_idx,keep_topK=200):
      start_time = time.time()
      indexes,images,lbls_smp,msks_smp,boxes         = batch

      # images                                        = images.to(device)
      # Load RPNs
      # Get proposals and features
      with torch.no_grad(): 
        
        #Should replace this with RPN
        proposals, roi_align_proposals ,regressor_labels , regressor_target,fpn_feat_list = get_inpts(images.to(device),lbls_smp,boxes,rpn,backbone,box,keep_topK=800)
        clas_logits,  box_pred                                                            = box(roi_align_proposals.detach().to(device))
        # Get ground truth
        boxes, scores, labels, gt_masks                                                   = self.preprocess_ground_truth_creation(indexes,clas_logits,box_pred,lbls_smp,proposals,\
                                                                                                              boxes,msks_smp,fpn_feat_list)
        # ROI Align 
        roi_align_proposals                            = self.MultiScaleRoiAlign(fpn_feat_list, boxes)
      # Forward function
      mask_pred                                      = self(roi_align_proposals)
      # Loss
      loss                                          = self.compute_loss(torch.vstack(mask_pred),torch.cat(labels),torch.vstack(gt_masks))
      return {'loss':loss}


    def validation_step(self, batch, batch_idx,keep_topK=200):
      indexes,images,lbls_smp,msks_smp,boxes         = batch

      images                                        = images.to(device)
      # Load RPNs
      # Get proposals and features
      with torch.no_grad(): 
        
        #Should replace this with RPN
        proposals, roi_align_proposals ,regressor_labels , regressor_target,fpn_feat_list = get_inpts(images.to(device),lbls_smp,boxes,rpn,backbone,box,keep_topK=800)
        clas_logits,  box_pred                                                            = box(roi_align_proposals.detach().to(device))
        # Get ground truth
        boxes, scores, labels, gt_masks                                                   = self.preprocess_ground_truth_creation(indexes,clas_logits,box_pred,lbls_smp,proposals,\
                                                                                                                                  boxes,msks_smp,fpn_feat_list)

        # ROI Align 
        roi_align_proposals                            = self.MultiScaleRoiAlign(fpn_feat_list, boxes)

      # Forward function
      mask_pred                                      = self(roi_align_proposals)
      # Loss
      loss                                          = self.compute_loss(torch.vstack(mask_pred),torch.cat(labels),torch.vstack(gt_masks))

      return {'val_loss':loss}

    def configure_optimizers(self):
      optim = torch.optim.Adam(self.parameters(), lr=0.001)
      lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [10, 15], gamma=0.1)
      return {"optimizer": optim, "lr_scheduler": lr_scheduler}


    def training_epoch_end(self, outputs):

      avg_train_loss        = torch.stack([x['loss'] for x in outputs]).mean().item()

      self.train_loss.append(avg_train_loss)

      print("train_loss", self.train_loss)
      self.log("train_loss", avg_train_loss,on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, validation_step_outputs):
      avg_val_loss        = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean().item()
      
      self.val_loss.append(avg_val_loss)

      print("val_loss", self.val_loss)

      self.log("val_loss", avg_val_loss,on_epoch=True, prog_bar=True)
