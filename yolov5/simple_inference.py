import torch
import torch.nn as nn
import cv2
import numpy as np
from utils import non_max_suppression, xywh2xyxy
from models.yolo import Model
from metrics import ConfusionMatrix, box_iou
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='weights/yolov5x.pt', help='model.pt path')
parser.add_argument('--image', type=str, default='inference/images/test.jpg', help='Input image') 
parser.add_argument('--output_dir', type=str, default='inference/output/', help='output directory')
parser.add_argument('--thres', type=float, default=0.4, help='object confidence threshold')
opt = parser.parse_args()


''' 
Class Labels 
Num : 80
'''

classnames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']


label = {}
for i, name in enumerate(classnames):
    label[i]=name



# load pre-trained model
weights = opt.weights

# try:
model = torch.load(weights)['model'].float()
model.eval()

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)




# except:
#     print('[ERROR] check the model')


def image_loader(im,imsize):
    '''
    processes input image for inference 
    '''
    h, w = im.shape[:2]
    im = letterbox(im, (640, 640), stride=32)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im)
    im = im.float()
    im /= 255.0
    im = im.unsqueeze(0)

    return im, h, w 

def scale_coords(old_shape, new_shape, coords):
  h, w = old_shape
  newH, newW = new_shape

  gainH = newH / h
  gainW = newW / w
  pad = (newW - w * gainW) / 2, (newH - h * gainH) / 2  # wh padding

  coords[:, [0, 2]] -= pad[0]  # x padding
  coords[:, [1, 3]] -= pad[1]  # y padding

  coords[:, 0] /= gainW
  coords[:, 1] /= gainH
  coords[:, 2] /= gainW
  coords[:, 3] /= gainH

  coords[:, 0].clamp_(0, w)  # x1
  coords[:, 1].clamp_(0, h)  # y1
  coords[:, 2].clamp_(0, w)  # x2
  coords[:, 3].clamp_(0, h)  # y2

  coords = coords.detach().numpy()

  return coords
    

def get_pred(img):
    '''
    returns prediction in numpy array
    '''
    imsize = 640
    img, h, w = image_loader(img,imsize)
    pred = model(img)[0]

    pred = non_max_suppression(pred, conf_thres=0.40) # conf_thres is confidence thresold

    if pred[0] is not None:
        _, newH, newW = img[0].shape
        gainH = newH / h
        gainW = newW / w
        pad = (newW - w * gainW) / 2, (newH - h * gainH) / 2  # wh padding
        pred = pred[0]

        pred[:, [0, 2]] -= pad[0]  # x padding
        pred[:, [1, 3]] -= pad[1]  # y padding

        pred[:, 0] /= gainW
        pred[:, 1] /= gainH
        pred[:, 2] /= gainW
        pred[:, 3] /= gainH

        pred[:, 0].clamp_(0, w)  # x1
        pred[:, 1].clamp_(0, h)  # y1
        pred[:, 2].clamp_(0, w)  # x2
        pred[:, 3].clamp_(0, h)  # y2

        pred = pred.detach().numpy()
    

    return pred
                

path = opt.image

def yolov5(img):
  #image = cv2.imread(path)
  image = img

  if image is not None:
      prediction = get_pred(image)

      if prediction is not None:
          for pred in prediction:

              x1 = int(pred[0])
              y1 = int(pred[1])
              x2 = int(pred[2])
              y2 = int(pred[3])

              start = (x1,y1)
              end = (x2,y2)

              pred_data = f'{label[pred[-1]]} {str(pred[-2]*100)[:5]}%'
              print(pred_data)
              color = (0,255,0)
              image = cv2.rectangle(image, start, end, color)
              image = cv2.putText(image, pred_data, (x1,y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA) 
          #cv2.imwrite(opt.output_dir+'result.jpg', image)

  else:
      print('[ERROR] check input image')

  return image


classes = {
  "car":2,
  "truck":7,
  "bus":5,
  "motorcycle":3
}

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def normyolo2xyxy(yolo_coords, w, h):
    # denormalize
    yolo_coords[:, [0, 2]] *= w
    yolo_coords[:, [1, 3]] *= h

    x1 = yolo_coords[:, 0] - yolo_coords[:, 2] / 2
    y1 = yolo_coords[:, 1] - yolo_coords[:, 3] / 2
    x2 = yolo_coords[:, 0] + yolo_coords[:, 2] / 2
    y2 = yolo_coords[:, 1] + yolo_coords[:, 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=1)

def coco2xyxy(x):
  y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
  y[:, 2] += x[:, 0]
  y[:, 3] += x[:, 1]
  return y



if __name__ == '__main__':
  # load image from coco dataset
  img_path = "000000015338.jpg"
  img = cv2.imread(img_path)
  image = img.copy()

  # get labels
  labels = [[5, 264.58,201.17,84.19,59.23],
            [2, 140.15,208.61,50.17,13.60],
            [2, 0.0,187.2,32.65,27.37],
            [5, 444.95,171.72,91.05,100.05],
            [7, 67.58,185.79,78.09,35.86]]
  
  # get detections
  detections = get_pred(img)

  detections = torch.tensor(detections)
  torch.set_printoptions(sci_mode=False)

  
  detections = torch.tensor(detections)
  labels = torch.tensor(labels)

  labels[:, 1:] = coco2xyxy(labels[:, 1:])
  iou = torch.linspace(0.5, 0.95, 10)

  # evaluate
  correct = process_batch(detections, labels, iou)
  print(detections)
  print(labels)
  print(correct)
  exit(0)

  for pred in labels[:, 1:]:
    x1 = int(pred[0])
    y1 = int(pred[1])
    x2 = int(pred[2])
    y2 = int(pred[3])

    start = (x1,y1)
    end = (x2,y2)

    #pred_data = f'{label[pred[-1]]} {str(pred[-2]*100)[:5]}%'
    #print(pred_data)
    color = (0,255,0)
    image = cv2.rectangle(image, start, end, color)
    image = cv2.putText(image, "", (x1,y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA) 
  cv2.imshow('frame', image)
  cv2.waitKey(0)


  """
  for pred in detections:
    x1 = int(pred[0])
    y1 = int(pred[1])
    x2 = int(pred[2])
    y2 = int(pred[3])

    start = (x1,y1)
    end = (x2,y2)

    #pred_data = f'{label[pred[-1]]} {str(pred[-2]*100)[:5]}%'
    #print(pred_data)
    color = (0,255,0)
    image = cv2.rectangle(image, start, end, color)
    image = cv2.putText(image, "", (x1,y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA) 
  cv2.imshow('frame', image)
  cv2.waitKey(0)
  """

  
  #cm = ConfusionMatrix(nc=7)
  #cm.process_batch(detections=detections, labels=labels)

  #print(detections)
  #print()
  #print(labels)



  """
  img = yolov5(img)
  print(img.shape)
  cv2.imshow('frame', img)
  cv2.waitKey(0) 
  """
