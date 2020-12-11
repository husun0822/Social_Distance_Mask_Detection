import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import torch
import torch.nn as nn
import os
import torchvision
import utils
import transforms as T

from PIL import Image
from torchvision import datasets
from facenet_pytorch import MTCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.nn import Softmax, Dropout, Linear
from torchvision import models
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor, CenterCrop, Normalize


class Person(object):
  def __init__(self, box, mask, frame, score, label, model_face, 
               model_mtcnn=model_face_mtcnn, model_mask=[model_mask_1,model_mask_2], frame_orig=None):
    """
    Inputs:
      box: (4,) ndarray, [xmin, ymin, xmax, ymax], in the entire frame
           x: ---> (horizontal, left: 0)
           y: |
              |
              | (vertical, top: 0)
      
      mask: (N, H) ndarray
      frame: ndarray, the entire image, of shape (H, W, C) 
      score: score from the human detection model
      label: 0: masked, 1: unmasked
      model_face: face detection model    
    """
    # plt.imshow(frame)
    # plt.show()
    self.box = box
    # print(box)
    self.mask_orig = mask
    _, self.mask = cv.threshold((mask * 255).astype(np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    self.mask = (self.mask / 255).astype(np.float32)
    self.frame = frame
    self.frame_orig = frame_orig
    self.score = score
    self.label = label
    self.pos = np.array([(self.box[0] + self.box[2]) / 2, self.box[3]])
    self.model_face = model_face
    self.model_mtcnn = model_mtcnn
    self.model_mask = model_mask
    self.face_loc = None  # global face location
    # self.prob = None
    self.face_loc_slice = None  # local face location
    self.face_loc_slice, self.face_loc = self.detect_face()
    self.prob_mask = self.detect_mask()

  def predict_(self, img, mask, anchor_x, anchor_y):
    """
    Inputs:
      img, mask: ndarray, (H, W, C)
    Outputs:
      x, y: float
    """
    self.model_face.eval()
    with torch.no_grad():
      img = img.transpose(2, 0, 1).astype(np.float32)
      mask_all = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
      for c in range(mask_all.shape[-1]):
        mask_all[:, :, c] = mask
      mask = mask_all.transpose(2, 0, 1).astype(np.float32)
      img_T = torch.tensor(img, device=device).unsqueeze(0)
      mask_T = torch.tensor(mask, device=device).unsqueeze(0)

      pred = self.model_face(img_T, mask_T)
      pred_x, pred_y = pred[:, 0, ...], pred[:, 1, ...]

      return pred_x.squeeze().item() + anchor_x, pred_y.squeeze().item() + anchor_y

  def predict_raw_img_(self, img, mask, H_all=248, W_all=173, if_plot=False):
    """
    Inputs:
      img, mask: ndarray, dtype is np.float32
    Outputs:
      x1, y1, x2, y2: int, locs of the bounding box
    """
    H, W = img.shape[:2]
    anchor_x, anchor_y = W_all / 2, H_all / 2
    img_resized = cv.resize(img, (W_all, H_all))
    mask_resized = cv.resize(mask, (W_all, H_all))
    pred_x, pred_y = self.predict_(img_resized, mask_resized, anchor_x, anchor_y)
    x, y = pred_x / W_all * W, pred_y / H_all * H
    y = int(y)

    # mask_gray = (cv.cvtColor(mask, cv.COLOR_BGR2GRAY) * 255).astype(np.uint8)
    mask_gray = (mask * 255).astype(np.uint8)
    _, mask_th = cv.threshold(mask_gray[:y, :], 0, 255, 
                              cv.THRESH_BINARY + cv.THRESH_OTSU)
    # mask_upper = (mask_th[:y, :] / 255).astype(np.float32)
    mask_upper = (mask_th / 255).astype(np.float32)
    # plt.imshow(mask_upper, cmap="gray")
    # plt.show()
    inds = np.argwhere(mask_upper > 0)
    yy, xx = inds[:, 0], inds[:, 1]
    # for x_iter in xx:
    #   print(x_iter)
    # print(xx.min())
    y_min, y_max = yy.min(), y
    x_min, x_max = xx.min(), xx.max()

    if if_plot:
      # plt.imshow(mask_th, cmap="gray")
      # plt.show()
      img_c = img.copy()
      img_c = cv.rectangle(img_c, (x_min, y_min), (x_max, y_max), 
                          color=(0, 0, 255), thickness=2)
      plt.imshow(cv.cvtColor(img_c, cv.COLOR_BGR2RGB))
      plt.show()

    return x_min, y_min, x_max, y_max
  

  def IOU_(self, box1, box2):
    if box1[2] < box2[0] or box2[2] < box1[0]:
      return 0
    elif box1[3] < box2[1] or box2[3] < box1[1]:
      return 0
    xx = sorted([box1[0], box1[2], box2[0], box2[2]])
    yy = sorted([box1[1], box1[3], box2[1], box2[3]])
    intersection = (xx[2] - xx[1]) * (yy[2] - yy[1])
    union = abs((box1[0] - box1[2]) * (box1[1] - box1[3])) + \
    abs((box2[0] - box2[2]) * (box2[1] - box2[3])) - intersection
    return intersection / union
  
  def detect_face(self, if_plot=True, th=0.35, tol=0.15, tol_half=0.4):
    """
    Inputs:
      model_face: face detection model
      offset: (2,) ndarray, (xmin, ymin) of the large bounding box(in the entire frame)
      if_plot: bool, whether to plot the person, including an original slice, a slice with face detection, 
               and the mask

    Outputs:
      box: (4,) ndarray, face loc in the person slice image
      face_loc: (4,) ndarray, [xmin, ymin, xmax, ymax] in the entire scene (offset has been added)
      prob: (1,) ndarray, the probability of the face
    """
    model_face = self.model_face
    box = self.box.astype(np.int)
    offset = box[:2]
    slice_fr = self.frame[box[1] : box[3], box[0] : box[2], :]
    slice_mask = self.mask_orig[box[1] : box[3], box[0] : box[2]]
    x_min, y_min, x_max, y_max = self.predict_raw_img_(slice_fr, slice_mask)
    local_face = np.array([x_min, y_min, x_max, y_max])
    global_face = local_face + np.array([offset[0], offset[1], offset[0], offset[1]])

    slice_fr = self.frame[box[1] : box[3], box[0] : box[2], :].transpose((2, 0, 1))
    tensor = torch.tensor(slice_fr)

    with torch.no_grad():
      trans = torchvision.transforms.ToPILImage()
      img_PIL = trans(tensor)

      # img_plot = np.array(img_PIL)
      # plt.imshow(img_plot)
      # plt.show()

      box, prob = None, None
      try:
        box, prob = self.model_mtcnn.detect(img_PIL)
      except RuntimeError as e:
        print("Runtime error!")
        img_plot = np.array(img_PIL)
        plt.imshow(img_plot)
        plt.show()
        return local_face, global_face

      #print(f"box: {box}, prob: {prob}")
      if prob is None or box is None:
        return local_face, global_face
      else:
        ind = np.argmax(prob)
        # box = box[ind : ind + 1, :]
        # prob = prob[ind : ind + 1]
        box = box[ind, :]
        prob = prob[ind]
        if prob < th:
          return local_face, global_face
        # elif box[1] >= local_face[3] - (local_face[3] - local_face[1]) * tol:
        elif self.IOU_(box, local_face) < tol:
          return local_face, global_face
        elif box[3] < local_face[3] - (local_face[3] - local_face[1]) * tol_half:
          return local_face, global_face
        else:
          face_loc = box + np.array([offset[0], offset[1], offset[0], offset[1]])
          return box, face_loc
    
  def detect_mask(self):
    # detect whether the face is wearing a mask
    box = self.box.astype(np.int)
    people_size = abs(box[2]-box[0])*abs(box[3]-box[1])
    # model_mask.eval()
    # prepare face slice data
    global_face_loc = (self.face_loc).astype(np.int)
    box = global_face_loc
    face_size = abs(box[2]-box[0])*abs(box[3]-box[1])
    # if (face_size/people_size > 1):
    #  model_mask = self.model_mask[1]
    #  use_soft = True
    # else:
    #   model_mask = self.model_mask[0]
    #  use_soft = False
    # print(global_face_loc)
    img = self.frame_orig
    face_slice = img[global_face_loc[1]:global_face_loc[3], global_face_loc[0]:global_face_loc[2],:]
    # face_slice = np.floor(face_slice * 255).astype(np.uint8)
    face_slice = face_slice.astype(np.uint8)

    size_grads = [14,28,56,112,224]
    def argmin_dist(img_size) -> int:
      H, W = img_size
      min_dist = 2 ** 32
      out_size = None

      for size_cand in size_grads:
        cur_dist = abs(H - size_cand) + abs(W - size_cand)
        if min_dist > cur_dist:
          out_size = size_cand
          min_dist = cur_dist
      
      return out_size

    if min(face_slice.shape[:2]) < 40:
      out_size = 4*argmin_dist(face_slice.shape[:2])
      model_mask = self.model_mask[1]
    else:
      out_size = argmin_dist(face_slice.shape[:2])
      model_mask = self.model_mask[0]

    transformations = T.Compose([
      T.ToPILImage(),
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      T.Resize((out_size, out_size))
    ])
    
    # transformations = Compose([
    #        ToPILImage(),
    #        Resize(256),
    #        CenterCrop(224),
    #        ToTensor(),
    #        Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    # ])
    # soft = Softmax()
    # if use_soft:
    #  prob_mask = soft(model_mask(transformations(face_slice).unsqueeze(0).to(device)))[0,1] # probability of wearing a mask
    #else:
    soft = nn.Softmax(dim=1)
    prob_mask = soft(model_mask(transformations(face_slice).unsqueeze(0).to(device)))[0,1]
    prob_mask = prob_mask.data.cpu().numpy()
    #print(prob_mask)
    return prob_mask


class Frame(object):
  # This class is for video analysis
  def __init__(self, img, label, model_detect, model_face, th1=0.5, frame_orig=None):
    """
    Inputs:
      img: ndarray, input frame
    """
    self.frame = img
    self.frame_orig = frame_orig
    self.label = label
    self.th1 = th1
    self.model_detect = model_detect
    self.model_face = model_face
    self.people = self.detect_people(False)
    self.frame_out = None
    self.mask_score = []
    self.face_proportion = []

  
  def detect_people(self, if_plot=True):
    model = self.model_detect
    model.eval()
    with torch.no_grad():
      img = torch.tensor(self.frame.transpose(2, 0, 1))
      preds = model([img.to(device)])
      boxes = preds[0]["boxes"].cpu()
      masks = preds[0]["masks"].cpu()
      scores = preds[0]["scores"].cpu()
      inds = (scores > self.th1)
      boxes = boxes[inds].numpy()
      masks = masks[inds].numpy()
      scores = scores[inds].numpy()
      img = img.numpy().transpose((1, 2, 0))

      num_people = scores.shape[0]
      people = []
      for i in range(num_people):
        person = Person(boxes[i], masks[i, 0], img, scores[i], self.label, model_face,frame_orig=self.frame_orig)
        # if person.prob is not None:
        people.append(person)
          # person.detect_face()
          # print(f"pos: {person.pos}")
      
      if if_plot:
        self.frame_out = self._draw()
      
      return people
    
  def _draw(self, if_plot=False):
      img_c = (self.frame.copy() * 255).astype(np.uint8)
      # plt.imshow(img_c)
      # plt.show() 

      for person in self.people:
        # draw on the frame

        # img_c = self.frame.copy()
        box = person.box.astype(np.int)
        img_c = cv.rectangle(img_c, (box[0], box[1]), (box[2], box[3]), 
                            color=(0, 255, 0), thickness=2)
        # img_c = cv.putText(img_c, f"score: {person.score: .3f}", 
        #                   (box[0], box[1]), cv.FONT_HERSHEY_SIMPLEX, 0.3, 
        #                   color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
        people_size = abs(box[2]-box[0])*abs(box[3]-box[1])

        if person.face_loc_slice is not None:
          # print(f"{person.face_loc_slice}, {person.face_loc}")
          # box = person.face_loc[0].astype(np.int)
          box = person.face_loc.astype(np.int)
          # prob = person.prob[0]
          wearing_mask_prob = person.prob_mask
          if wearing_mask_prob > 0.5:
            img_c = cv.rectangle(img_c, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=2)
            img_c = cv.putText(img_c, f"{wearing_mask_prob: .3f}", (box[0], box[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, 
                               color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
          else:
            img_c = cv.rectangle(img_c, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=2)
            img_c = cv.putText(img_c, f"{wearing_mask_prob: .3f}", (box[0], box[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, 
                               color=(255, 0, 0), thickness=2, lineType=cv.LINE_AA)
          
          # calculate the proportion of bounding box
          face_size = abs(box[2]-box[0])*abs(box[3]-box[1])
          face_proportion = face_size/people_size
          #img_c = cv.putText(img_c, f"{face_proportion: .3f}", (box[0], box[3]+10), cv.FONT_HERSHEY_SIMPLEX, 0.5, 
          #                     color=(255,0,255), thickness=2, lineType=cv.LINE_AA)

          # ### modified: no prob printed
          # img_c = cv.putText(img_c, f"prob: {prob: .3f}", (box[0], box[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, 
          #                     color=(0, 255, 255), thickness=2, lineType=cv.LINE_AA)
          # ### end modified
            
          # print(f"img_c's type: {type(img_c)}")
          # img_c = cv.UMat.get(img_c)
          self.mask_score.append(wearing_mask_prob)
          self.face_proportion.append(face_proportion)
        
      if if_plot:
        plt.imshow(img_c)
        plt.show()

      return (img_c / 255).astype(np.float32)
      
