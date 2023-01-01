import numpy as np
import torch
from functools import partial

def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

def visualize_anchors_one_img(img, bbox, anch):
  # Anchors are in N,4
  # img in 1,3,800,1088
  fix,ax  = plt.subplots(1,1)
  ax.imshow(img.squeeze(0).permute(1,2,0))
    
  for pred in anch:
    col   = 'b'
    w     = pred[2]  - pred[0]
    h     = pred[3]  - pred[1]
    rect  = patches.Rectangle((pred[0],pred[1]),w,h,fill=False,linewidth = 1,color=col)
    ax.add_patch(rect)
  

  for box in bbox:
    col   = 'r'
    box   = box.flatten()
    w     = box[2]  - box[0]
    h     = box[3]  - box[1]
    rect  = patches.Rectangle((box[0],box[1]),w,h,fill=False,linewidth = 3,color=col)
    ax.add_patch(rect)
  
  plt.show()

'''
BoxHead helpers
'''
import matplotlib.lines as mlines
'''
Pretrained RPN
'''
def pretrained_models_680(checkpoint_file,eval=True):
    import torchvision
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)

    if(eval):
        model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    backbone = model.backbone
    rpn = model.rpn

    if(eval):
        backbone.eval()
        rpn.eval()

    rpn.nms_thresh=0.6
    checkpoint = torch.load(checkpoint_file,map_location=device)

    backbone.load_state_dict(checkpoint['backbone'])
    rpn.load_state_dict(checkpoint['rpn'])

    return backbone, rpn
'''
Gets proposals, ROI, gt_label and gt_target 
'''
def get_inpts(images, gt_labels, bbox,rpn,backbone,box,keep_topK=200):
  # Inputs are batch from dataloader, pretrained rpn object and backbone.
  # Take the features from the backbone
  backout   = backbone(images)
  # The RPN implementation takes as first argument the following image list
  im_lis    = ImageList(images, [(800, 1088)]*images.shape[0])
  # Then we pass the image list and the backbone output through the rpn
  rpnout    = rpn(im_lis, backout)

  #The final output is
  # A list of proposal tensors: list:len(bz){(keep_topK,4)}
  proposals     = [proposal[0:keep_topK,:] for proposal in rpnout[0]]
  # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
  fpn_feat_list = list(backout.values())

  '''
  Example on how to call ROI -- roi below would be of shape (num_proposals,256*P^2) => (2000,12544)
  '''
  roi_align_proposals               = box.MultiScaleRoiAlign(fpn_feat_list,proposals)
  gt_class, gt_regressor_target     = box.create_ground_truth(proposals,gt_labels,bbox)


  return proposals, roi_align_proposals, gt_class, gt_regressor_target,fpn_feat_list

  

def plot_ground_truth_assignment(images, labels,regressor_target,proposals, keep_topK=20):
  # # Visualization of the proposals
  i =0
  color_dict  =  {1:'r',2:'b',3:'g' }
  color_dark  =  {1:'darkred',2:'darkblue',3:'darkgreen' }
  vehicle_gt = mlines.Line2D([], [], color='darkred', ls='-', label='vehicle_gt')
  person_gt = mlines.Line2D([], [], color='darkblue', ls='-', label='person_gt')
  animal_gt = mlines.Line2D([], [], color='darkgreen', ls='-', label='animal_gt')
  vehicle = mlines.Line2D([], [], color='red', ls='-', label='vehicle')
  person = mlines.Line2D([], [], color='blue', ls='-', label='person')
  animal = mlines.Line2D([], [], color='green', ls='-', label='animal')


  img_squeeze = transforms.functional.normalize(images[i,:,:,:].to('cpu'),
                                                [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
  fig,(ax1,ax2)=plt.subplots(1,2, figsize=(16, 16))

  ax1.imshow(img_squeeze.permute(1,2,0))
  for encoded_label, regress_encoded_target, proposal_batch in zip(labels[keep_topK*i:keep_topK*i+keep_topK],regressor_target[keep_topK*i:keep_topK*i+keep_topK], proposals[i]):
      if(encoded_label!=0):
        target_box = box.decode_target(proposal_batch,regress_encoded_target).cpu()
        rect=patches.Rectangle((target_box[0] - target_box[2]/2,target_box[1] - target_box[3]/2),target_box[2],target_box[3],fill=False,color=color_dark[encoded_label.item()],lw=4)
        ax1.add_patch(rect)

  ax1.legend(handles=[vehicle_gt, person_gt,animal_gt])
  ax1.set_title("Ground truth")

  ax2.imshow(img_squeeze.permute(1,2,0))
  for encoded_label, regress_encoded_target, proposal_batch in zip(labels[keep_topK*i:keep_topK*i+keep_topK],regressor_target[keep_topK*i:keep_topK*i+keep_topK], proposals[i].cpu()):
      if(encoded_label!=0):
        rect=patches.Rectangle((proposal_batch[0],proposal_batch[1]),proposal_batch[2]-proposal_batch[0],proposal_batch[3]-proposal_batch[1],fill=False,color=color_dict[encoded_label.item()])
        ax2.add_patch(rect)
  ax2.legend(handles=[vehicle, person,animal])
  ax2.set_title("Top 20 Proposals")
  plt.show()


def plot_top20_proposals(images, labels,regressor_target,proposals,pre_boxes, pre_labels, keep_topK=20):
  # # Visualization of the proposals
  i =0
  b = 0
  color_dict  =  {0:'r',1:'b',2:'g' }
  color_dark  =  {1:'darkred',2:'darkblue',3:'darkgreen' }
  vehicle_gt = mlines.Line2D([], [], color='darkred', ls='-', label='vehicle_gt')
  person_gt = mlines.Line2D([], [], color='darkblue', ls='-', label='person_gt')
  animal_gt = mlines.Line2D([], [], color='darkgreen', ls='-', label='animal_gt')
  vehicle = mlines.Line2D([], [], color='red', ls='-', label='vehicle')
  person = mlines.Line2D([], [], color='blue', ls='-', label='person')
  animal = mlines.Line2D([], [], color='green', ls='-', label='animal')


  img_squeeze = transforms.functional.normalize(images[i,:,:,:].to('cpu'),
                                                [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
  fig,(ax1,ax2)=plt.subplots(1,2, figsize=(16, 16))

  ax1.imshow(img_squeeze.permute(1,2,0))
  for encoded_label, regress_encoded_target, proposal_batch in zip(labels[keep_topK*i:keep_topK*i+keep_topK],regressor_target[keep_topK*i:keep_topK*i+keep_topK], proposals[i]):
      if(encoded_label!=0):
        target_box = box.decode_target(proposal_batch,regress_encoded_target).cpu()
        rect=patches.Rectangle((target_box[0] - target_box[2]/2,target_box[1] - target_box[3]/2),target_box[2],target_box[3],fill=False,color=color_dark[encoded_label.item()],lw=4)
        ax1.add_patch(rect)
  ax1.legend(handles=[vehicle_gt, person_gt,animal_gt])
  ax1.set_title("Ground truth")


  ax2.imshow(img_squeeze.permute(1,2,0))
  plt_box                   = pre_boxes[b]
  plt_labels                = pre_labels[b]
  for idx,bbox_plt in enumerate(plt_box):
    bbox_plt                     = bbox_plt.cpu()
    lbl                     = plt_labels[idx].cpu()
    rect=patches.Rectangle((bbox_plt[0] ,bbox_plt[1]),bbox_plt[2] - bbox_plt[0] ,bbox_plt[3] - bbox_plt[1],fill=False,color=color_dict[lbl.item()],lw=2)
    ax2.add_patch(rect)
  ax2.legend(handles=[vehicle, person,animal])
  ax2.set_title("Top 20 Proposals predicted by Box Head")
  plt.show()


def plot_pre_post_nms(images,bbox, gt_labels,  pre_boxes, pre_labels, boxes, labels, keep_topK=20):
  # # Visualization of the proposals
  b =0
  color_dict  =  {0:'r',1:'b',2:'g' }
  color_dark  =  {0:'darkred',1:'darkblue',2:'darkgreen' }
  vehicle_gt = mlines.Line2D([], [], color='darkred', ls='-', label='vehicle_gt')
  person_gt = mlines.Line2D([], [], color='darkblue', ls='-', label='person_gt')
  animal_gt = mlines.Line2D([], [], color='darkgreen', ls='-', label='animal_gt')
  vehicle = mlines.Line2D([], [], color='red', ls='-', label='vehicle')
  person = mlines.Line2D([], [], color='blue', ls='-', label='person')
  animal = mlines.Line2D([], [], color='green', ls='-', label='animal')

  fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(30, 48))

  img_squeeze = transforms.functional.normalize(images[b,:,:,:].to('cpu'),
                                      [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                      [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
  # Ground truth
  ax1.imshow(img_squeeze.permute(1,2,0))
  plt_box                   = bbox[b]
  plt_labels                = gt_labels[b]
  for idx,bbox_plt in enumerate(plt_box):
    bbox_plt                = bbox_plt.cpu()
    lbl                     = int(plt_labels[idx].cpu()) - 1
    rect=patches.Rectangle((bbox_plt[0] ,bbox_plt[1]),bbox_plt[2] - bbox_plt[0] ,bbox_plt[3] - bbox_plt[1],fill=False,color=color_dark[lbl],lw=2)
    ax1.add_patch(rect)
  ax1.legend(handles=[vehicle_gt, person_gt,animal_gt])
  ax1.set_title("Ground truth")
  
  # Pre-NMS
  ax2.imshow(img_squeeze.permute(1,2,0))
  plt_box                   = pre_boxes[b]
  plt_labels                = pre_labels[b]
  for idx,bbox_plt in enumerate(plt_box):
    bbox_plt                     = bbox_plt.cpu()
    lbl                     = plt_labels[idx].cpu()
    rect=patches.Rectangle((bbox_plt[0] ,bbox_plt[1]),bbox_plt[2] - bbox_plt[0] ,bbox_plt[3] - bbox_plt[1],fill=False,color=color_dict[lbl.item()],lw=2)
    ax2.add_patch(rect)
  ax2.legend(handles=[vehicle, person,animal])
  ax2.set_title("Pre-NMS")

  # Post NMS
  ax3.imshow(img_squeeze.permute(1,2,0))
  plt_box                   = boxes[b]
  plt_labels                = labels[b]
  for idx,bbox_plt in enumerate(plt_box):
    if(len(bbox_plt) == 0):
      continue
    bbox_plt                     = bbox_plt.cpu()
    lbl                          = idx
    for nms_bbox in bbox_plt:
      
      rect=patches.Rectangle((nms_bbox[0] ,nms_bbox[1]),nms_bbox[2] - nms_bbox[0] ,nms_bbox[3] - nms_bbox[1],fill=False,color=color_dict[lbl],lw=2)
      ax3.add_patch(rect)
  ax3.legend(handles=[vehicle, person,animal])
  ax3.set_title("Post-NMS")
  plt.show()

def concat(boxes,scores,labels):
  # boxes -> list(len(bz){list(class_0,class_1,class_2)})
  ret_boxes   = []
  ret_labels  = []
  ret_scores  = []
  bz      = len(boxes)

  for b,l,s in zip(boxes,scores,labels):
    sub_b,sub_l,sub_s = [],[],[]
    for s_b,s_l,s_s in zip(b,l,s):
      if( len(s_b)!=0 ):
        sub_b.append(s_b)
        sub_l.append(s_l)
        sub_s.append(s_s)
    ret_boxes.append(torch.vstack(sub_b))
    ret_labels.append(torch.cat(sub_l))
    ret_scores.append(torch.cat(sub_s))

  return ret_boxes,ret_labels,ret_scores

def plot_all(images,bbox,masks,labels):
  # # Visualization of the proposals
  b = 0
  color_dict  =  {0:'r',1:'b',2:'g' }
  color_dark  =  {0:'darkred',1:'darkblue',2:'darkgreen' }
  vehicle_gt = mlines.Line2D([], [], color='darkred', ls='-', label='vehicle_gt')
  person_gt = mlines.Line2D([], [], color='darkblue', ls='-', label='person_gt')
  animal_gt = mlines.Line2D([], [], color='darkgreen', ls='-', label='animal_gt')
  vehicle = mlines.Line2D([], [], color='red', ls='-', label='vehicle')
  person = mlines.Line2D([], [], color='blue', ls='-', label='person')
  animal = mlines.Line2D([], [], color='green', ls='-', label='animal')

  fig,ax1=plt.subplots(1,1,figsize=(12, 5))

  img_squeeze = transforms.functional.normalize(images[b,:,:,:].to('cpu'),
                                      [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                      [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
  # box
  ax1.imshow(img_squeeze.permute(1,2,0))
  plt_box                   = bbox[b]
  plt_labels                = labels[b]
  plt_masks                 = masks[b]
  for idx,bbox_plt in enumerate(plt_box):
    bbox_plt                = bbox_plt.cpu()
    lbl                     = int(plt_labels[idx].cpu())
    msk                     = 1*(plt_masks[b]>0.5)
    rect=patches.Rectangle((bbox_plt[0] ,bbox_plt[1]),bbox_plt[2] - bbox_plt[0] ,bbox_plt[3] - bbox_plt[1],fill=False,color=color_dark[lbl],lw=2)
    ax1.add_patch(rect)
    ax1.imshow(msk.squeeze(0).squeeze(0).detach().cpu().numpy(), cmap  = ListedColormap(['none',color_dark[lbl]]), alpha=0.5)
  ax1.legend(handles=[vehicle_gt, person_gt,animal_gt])

'''
Plot RPNS
'''
def rpn_plt(image,rpns,gt_box):
  '''
  images -> (3,800,1088) 
  gt_box  - >({num_obj,4}) (x1,y1,x2,y2)
  rpns    -> (num_proposals,4)
  '''
  img_squeeze = transforms.functional.normalize(image[0,:,:,:].to('cpu'),
                                    [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                    [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
  rpn = mlines.Line2D([], [], color='b', ls='-', label='rpn_proposals')
  gt = mlines.Line2D([], [], color='r', ls='-', label='ground_truth')
  fig,ax1=plt.subplots(1,1,figsize=(12, 5))
  ax1.imshow(img_squeeze.permute(1,2,0))
  for box in gt_box:
    box             = box.flatten().cpu()
    rect=patches.Rectangle((box[0],box[1]),box[2] - box[0] ,box[3] - box[1],fill=False,color='r',lw=3)
    ax1.add_patch(rect)

  for box in rpns:
    box = box.cpu()
    rect=patches.Rectangle((box[0] ,box[1] ),box[2] - box[0] ,box[3] - box[1],fill=False,color='b',lw=2)
    ax1.add_patch(rect)
  ax1.legend(handles=[gt,rpn])
  

def visualize_final(box,mask,batch):
  '''
  box   -> BoxHead object
  mask  -> MaskHead object
  batch -> Batch in testloader/trainloader
  '''
  vehicle_gt = mlines.Line2D([], [], color='darkred', ls='-', label='vehicle')
  person_gt = mlines.Line2D([], [], color='darkblue', ls='-', label='person')
  animal_gt = mlines.Line2D([], [], color='darkgreen', ls='-', label='animal')
  vehicle = mlines.Line2D([], [], color='red', ls='-', label='vehicle')
  person = mlines.Line2D([], [], color='blue', ls='-', label='person')
  animal = mlines.Line2D([], [], color='green', ls='-', label='animal')
  indexes,images,lbls_smp,msks_smp,boxes  = batch
  proposals, roi_align_proposals ,regressor_labels , regressor_target,fpn_feat_list = get_inpts(images.to(device),lbls_smp,boxes,rpn,backbone,box,keep_topK=500)
  clas_logits,  box_pred                                                            = box(roi_align_proposals.detach().to(device))
  softmax                                                                           = nn.Softmax(dim=1)
  clas_logits                                                                       = softmax(clas_logits)
  boxes, scores, labels, pre_boxes, pre_scores, pre_labels  = box.postprocess_detections(clas_logits,box_pred,proposals,keep_num_preNMS=800,keep_num_postNMS=1)
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

  color_dict  =  {0:'r',1:'b',2:'g' }
  color_dark  =  {0:'darkred',1:'darkblue',2:'darkgreen' }
  fig,ax1=plt.subplots(1,1,figsize=(12, 5))
  images = transforms.functional.normalize(images[0,:,:,:].to('cpu'),
                                    [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                    [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
  ax1.imshow(images.permute(1,2,0))

  for m,b,l in zip(msk_b_reshaped,box_b,labels_b):
    bbox_plt                = b.cpu()
    lbl                     = int(l.cpu())
    # To reshape msk to required 800,1088 bin image
    msk                 = torch.zeros(800,1088)
    msk[b[1].int():b[3].int(),b[0].int():b[2].int()]  = m
    # Now plot
    rect=patches.Rectangle((bbox_plt[0] ,bbox_plt[1]),bbox_plt[2] - bbox_plt[0] ,bbox_plt[3] - bbox_plt[1],fill=False,color=color_dark[lbl],lw=2)
    ax1.add_patch(rect)
    ax1.imshow(msk.squeeze(0).squeeze(0).detach().cpu().numpy(), cmap  = ListedColormap(['none',color_dark[lbl]]), alpha=0.5)
  ax1.legend(handles=[vehicle_gt, person_gt,animal_gt])
