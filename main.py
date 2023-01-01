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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from backbone import Resnet50Backbone
from utils import *
device      = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from pprint import pprint
SAVE_PATH = './' # Change this to wherever you want the checkpoint weights to be saved at

from rpn_head import RPNHead
from box_head import BoxHead
from mask_head import MaskHead
from pl_dataset import *

if __name__ == "__main__":
  '''
  Initialize dataset and datamodule
  '''
  imgs_path   = 'img_comp_zlib.h5'
  masks_path  = 'mask_comp_zlib.h5'
  labels_path = "labels_comp_zlib.npy"
  bboxes_path = "bboxes_comp_zlib.npy"

  paths = [imgs_path, masks_path, labels_path, bboxes_path]
  # load the data into data.Dataset
  dataset = BuildDataset(paths)

  # Standard Dataloaders Initialization
  batch_size    = 1
  datamodule    = rcnn_datamodule(dataset,batch_size=batch_size)
  datamodule.setup()

  '''
  RPN Training loop
  '''
  device      = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  # batch_size = 2
  # SGD --> lr =0.1, weight decay and LR Scheduler
  # max_epochs = 36

  # Make sure the paths are uncommented in the first cell to save the model checkpoints and losses
  model                 =   RPNHead().to(device)
  model_dir             = model.save_path+'rpn_checkpoints/'

  # DataLoader
  rcnn_lightning_datamodule = rcnn_datamodule(dataset,batch_size = 4)

  # Making all the callbacks -- checkpoints, learning rate monitor 
  checkpoint_callback   =   pl_callbacks.ModelCheckpoint(dirpath=model_dir)
  lr_monitor            =   LearningRateMonitor(logging_interval='step')
  callback              =   [checkpoint_callback, lr_monitor]
  tb_logger             =   pl_loggers.TensorBoardLogger("save_dir",name="rpn")

  trainer               =   pl.Trainer(gpus =1,logger=tb_logger, max_epochs=36, callbacks=callback)
  trainer.fit(model, rcnn_lightning_datamodule)

  model                 =   RPNHead()
  print(model)

  '''
  Testing anchors: Visualize anchors and ground truth
  '''

  rpn = RPNHead()
  anch = rpn.get_anchors()
  idx,img, label, mask, bbox = next(iter(datamodule.train_dataloader())) 
  label = list(label)
  mask = list(mask)
  bboxes_list = list(bbox)
  indexes = list(idx)
  image_shape = (800,1088)
  cls, coord = rpn.create_batch_truth(bboxes_list, indexes, image_shape)
  pred = rpn.decode_anch(cls,coord)
  print('bbox', bbox)
  print('pred', pred)
  visualize_anchors_one_img(img,bbox[0],pred)

  # Here we keep the top 20, but during training you should keep around 200 boxes from the 1000 proposals
  batch_size    = 1
  datamodule    = rcnn_datamodule(dataset,batch_size=batch_size)
  datamodule.setup()
  test_loader = datamodule.val_dataloader()
  pretrained_path='checkpoint680.pth'
  backbone, rpn = pretrained_models_680(pretrained_path)
  # # # keep_topK=200
  box       = BoxHead().to(device)
  box       = box.load_checkpoint().to(device)
  with torch.no_grad():
      for i, batch in enumerate(test_loader, 0):
          images    = batch[1].to(device)
          gt_labels = batch[2]
          bbox      = batch[4]
          
          proposals, roi_align_proposals ,regressor_labels , regressor_target,_ = get_inpts(images,gt_labels,bbox,rpn,backbone,box,keep_topK=800)
          # plot ground truth assignment
          # plot_ground_truth_assignment(images, regressor_labels,regressor_target,proposals, keep_topK=20)
          
          # 
          clas_logits,  box_pred      = box(roi_align_proposals.detach().to(device))
          softmax                     = nn.Softmax(dim=1)
          clas_logits                 = softmax(clas_logits)
          boxes, scores, labels, pre_boxes, pre_scores, pre_labels  = box.postprocess_detections(clas_logits,box_pred,proposals,keep_num_postNMS=1)
          
          # plot top 20 proposals 
          plot_top20_proposals(images, regressor_labels,regressor_target,proposals,pre_boxes, pre_labels, keep_topK=20)

          #plot pre and post nms results
          plot_pre_post_nms(images,bbox, gt_labels,  pre_boxes, pre_labels, boxes, labels, keep_topK=20)

          print('---------------------------------------------------------------------------------------------')
          print('---------------------------------------------------------------------------------------------')
        
          if i>3:
            break


  '''
  Ground truth creation
  '''
  batch_size    = 1
  datamodule    = rcnn_datamodule(dataset,batch_size=batch_size)
  datamodule.setup()
  test_loader = datamodule.val_dataloader()
  idx,images,gt_labels, transed_mask, bbox =  next(iter(test_loader))
  print("Original image")
  plt.imshow(images.squeeze().permute(1,2,0))
  plt.show()
  plt.close()
  box       = BoxHead().to(device)
  box       = box.load_checkpoint().to(device)
  # pretrained_path='checkpoint680.pth'
  # backbone, rpn = pretrained_models_680(pretrained_path)
  proposals, roi_align_proposals ,regressor_labels , regressor_target,fpn_feat_list = get_inpts(images.to(device),gt_labels,bbox,rpn,backbone,box,keep_topK=500)

  clas_logits,  box_pred      = box(roi_align_proposals.detach().to(device))
  mask  = MaskHead()
  boxes, scores, labels, gt_masks = mask.preprocess_ground_truth_creation(idx,clas_logits,box_pred,gt_labels,proposals,bbox,transed_mask,fpn_feat_list,debug=True)


  '''
  BOX Training loop
  '''
  box                     =  BoxHead().to(device)
  checkpoint_callback     =   pl_callbacks.ModelCheckpoint(dirpath=save_dir_checkpoint)
  lr_monitor              =   LearningRateMonitor(logging_interval='step')
  callback                =   [checkpoint_callback, lr_monitor]
  tb_logger               =   pl_loggers.TensorBoardLogger("save_dir",name="rcnn_b")

  # Training 
  trainer                 =   pl.Trainer(accelerator='gpu', devices=1,logger=tb_logger, max_epochs=40, callbacks=callback)
  trainer.fit(box, datamodule)



  '''
  MASK Training loop
  '''
  ######Replace this with implemented RPN head later
  pretrained_path='checkpoint680.pth'
  backbone, rpn = pretrained_models_680(pretrained_path)
  ###########################################
  box        = BoxHead().load_checkpoint().to(device)
  box.eval()
  device      = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  # batch_size = 2
  # SGD --> lr =0.1, weight decay and LR Scheduler
  # max_epochs = 36

  # Make sure the paths are uncommented in the first cell to save the model checkpoints and losses
  model                 = MaskHead().to(device)
  model_dir             = model.save_path+'mask_checkpoints/'

  # DataLoader
  rcnn_lightning_datamodule = rcnn_datamodule(dataset,batch_size = 1)

  # Making all the callbacks -- checkpoints, learning rate monitor 
  checkpoint_callback   =   pl_callbacks.ModelCheckpoint(dirpath=model_dir)
  lr_monitor            =   LearningRateMonitor(logging_interval='step')
  early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min")
  callback              =   [checkpoint_callback, lr_monitor,early_stop_callback]
  # callback              =   [lr_monitor]

  tb_logger             =   pl_loggers.TensorBoardLogger("save_dir",name="mask")
  trainer               =   pl.Trainer(gpus =1,logger=tb_logger, max_epochs=36, callbacks=callback)
  trainer.fit(model, rcnn_lightning_datamodule)

  pretrained_path='checkpoint680.pth'
  backbone, rpn = pretrained_models_680(pretrained_path)
  model_head  = RPNHead().load_checkpoint().to(device)
  box         = BoxHead().load_checkpoint().to(device)
  mask        = MaskHead().load_checkpoint().to(device)
  for i,batch in enumerate(datamodule.train_dataloader()):
    visualize_final(box,mask,batch)
    if(i>10):
      break

  '''
  RPN_train_curve 
  '''
  train = SAVE_PATH+'/RPN/train_loss.npy'
  val = SAVE_PATH+'/RPN/val_loss.npy'
  train_clas = SAVE_PATH+'/RPN/train_class_loss.npy'
  val_clas = SAVE_PATH+'/RPN/val_class_loss.npy'
  train_reg = SAVE_PATH+'/RPN/train_reg_loss.npy'
  val_reg = SAVE_PATH+'/RPN/val_reg_loss.npy'

  tr_loss = np.load(train)
  val_loss = np.load(val)
  tr_cls_loss = np.load(train_clas)
  val_cls_loss = np.load(val_clas)
  tr_reg_loss = np.load(train_reg)
  val_reg_loss = np.load(val_reg)

  fig, (ax1,ax2,ax3) = plt.subplots(1, 3,figsize=(15, 5))

  ax1.plot(tr_cls_loss,lw=3,label='train_class_loss')
  ax1.plot(val_cls_loss,lw=3,label='val_class_loss')
  ax1.set_title("Classification loss",fontsize=20)
  ax1.set_ylabel("loss",fontsize=20)
  ax1.set_xlabel("epoch(s)",fontsize=20)

  ax2.plot(tr_reg_loss,lw=3,label='train_reg_loss')
  ax2.plot(val_reg_loss,lw=3,label='val_reg_loss')
  ax2.set_title("Regression loss",fontsize=20)
  ax2.set_ylabel("loss",fontsize=20)
  ax2.set_xlabel("epoch(s)",fontsize=20)

  ax3.plot(tr_loss,lw=3,label='Train')
  ax3.plot(val_loss,lw=3,label='Val')
  ax3.set_title("Total loss",fontsize=20)
  ax3.set_ylabel("loss",fontsize=20)
  ax3.set_xlabel("epoch(s)",fontsize=20)
  plt.legend(loc='lower center', bbox_to_anchor=(-0.7,-0.28),ncol=2,fontsize=15)

  '''
  Mask_train_curve 
  '''
  train = SAVE_PATH+'/Mask/train_loss.npy'
  val = SAVE_PATH+'/Mask/val_loss.npy'

  tr_loss = np.load(train)
  val_loss = np.load(val)

  plt.plot(tr_loss,label='train_loss',lw=3)
  plt.plot(val_loss,label='val_loss',lw=3)
  plt.legend()
  plt.ylabel("loss",fontsize=20)
  plt.xlabel("epoch(s)",fontsize=20)
  plt.title("Mask Head Loss",fontsize=20
            )

  from torchmetrics.detection.mean_ap import MeanAveragePrecision
  true_bbox = []
  true_lbl  = []
  true_msks = []

  pred_bbox = []
  pred_lbl  = []
  pred_msks = []
  pred_scores = []
  pretrained_path='checkpoint680.pth'
  backbone, rpn = pretrained_models_680(pretrained_path)
  backbone  = backbone.to(device)
  rpn       = rpn.to(device)
  model_head  = RPNHead().load_checkpoint().to(device)
  box         = BoxHead().load_checkpoint().to(device)
  mask        = MaskHead().load_checkpoint().to(device)
  module = rcnn_datamodule(dataset,batch_size = 1)
  module.setup()
  test_loader = module.val_dataloader()

  for idx in range(20):
    batch                                   = next(iter(test_loader))
    idx,images,gt_labels, transed_mask, bbox =  batch
    # device            = torch.device('cpu')
    true_bbox.extend(list(bbox))
    true_lbl.extend(list(gt_labels))
    true_msks.extend(list(transed_mask))

    indexes,images,lbls_smp,msks_smp,boxes  = batch
    proposals, roi_align_proposals ,regressor_labels , regressor_target,fpn_feat_list = get_inpts(images.to(device),lbls_smp,boxes,rpn,backbone,box,keep_topK=500)
    clas_logits,  box_pred                                                            = box(roi_align_proposals.detach().to(device))
    softmax                                                                           = nn.Softmax(dim=1)
    clas_logits                                                                       = softmax(clas_logits)
    boxes, scores, labels, pre_boxes, pre_scores, pre_labels = box.postprocess_detections(clas_logits,box_pred,proposals,keep_num_preNMS=800,keep_num_postNMS=1)
    boxes,scores,labels         = concat(boxes,scores,labels)
    roi_align_proposals         = mask.MultiScaleRoiAlign(fpn_feat_list, boxes)
    mask_pred                   = mask(roi_align_proposals)
    # msk                         = mask[0][torch.arange(len(mask[0])), labels]
    box_b                       = boxes[0]
    labels_b                    = labels[0]
    msk = mask_pred[0]
    msk = msk[torch.arange(len(msk)),labels_b.long()]
    # get height and width of each bbox
    w                           = box_b[:,2].int()  - box_b[:,0].int()
    h                           = box_b[:,3].int()  - box_b[:,1].int()

    # reshape all mask predictions within bound box dims
    # This would be a list of binary masks -- each of {h_box,w_box}
    msk_b_reshaped              = list(map(lambda msk,h,w: nn.functional.interpolate(msk.unsqueeze(0).unsqueeze(0).float()\
                                              ,size=(h.int().item(),w.int().item()), mode='bilinear').squeeze()>0.5,msk,h,w))
    msk_plt                         = torch.zeros((len(msk_b_reshaped),800,1088))
    for idx,(m,b,l) in enumerate(zip(msk_b_reshaped,box_b,labels_b)):
      bbox_plt                = b.cpu()
      lbl                     = int(l.cpu())
      # To reshape msk to required 800,1088 bin image
      # msk                 = torch.zeros(800,1088)
      msk_plt[idx,b[1].int():b[3].int(),b[0].int():b[2].int()]  = m
    
    pred_bbox.extend(box_b)
    pred_lbl.extend(labels_b)
    pred_msks.extend(msk_plt.unsqueeze(0))
    pred_scores.extend(scores[0])


  '''
  Prints ground truth and predicted masks
  '''
  print(len(msks_smp))
  print(msk_plt.shape)
  idx = 0
  print(boxes[idx].shape)
  plt.imshow(torch.vstack(transed_mask)[idx].squeeze().cpu())
  plt.show()
  plt.imshow(msk_plt[idx].squeeze().cpu())


  target_dict       = {
                          'boxes': torch.vstack(true_bbox).to(device),
                          'labels':torch.cat(true_lbl).to(device) - 1,
                          'masks': torch.vstack(true_msks).to(device).type(torch.uint8)
                        }


  pred_dict       = {
                          'boxes': torch.vstack(pred_bbox).to(device),
                          'labels':torch.stack(pred_lbl).to(device),
                          # 'labels':torch.cat(gt_labels).to(device)[:len(bbox)] - 1,
                          'masks': torch.vstack(pred_msks).type(torch.uint8),
                          'scores':torch.stack(pred_scores).to(device)
                        }

  metric = MeanAveragePrecision(iou_type = "bbox")
  metric.update([pred_dict], [target_dict])
  print("Validation bounding box mAP")
  pprint(metric.compute())

  metric = MeanAveragePrecision(iou_type = "segm")
  metric.update([pred_dict], [target_dict])
  print("Validation mask mAP")
  pprint(metric.compute())


