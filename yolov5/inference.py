from tkinter import W
import torch
import torch.nn as nn
import cv2
import numpy as np
from utils.general import non_max_suppression, letterbox, image_loader
from utils.loss import ComputeLoss
from utils.metrics import ConfusionMatrix, process_batch, ap_per_class
from utils.conversions import coco2xyxy, scale_coords, normyolo2xyxy, clip_coords
from models.yolo import Model

classnames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

classnames = {k:classnames.index(k) for k in classnames}


label = {}
for i, name in enumerate(classnames):
  label[i]=name



# load pre-trained model
weights = 'weights/yolov5x.pt'

model = torch.load(weights)['model'].float()
model.eval()


def get_pred(img):
  '''
  returns prediction in numpy array
  '''
  imsize = 640
  img, h, w = image_loader(img,imsize)
  pred = model(img)[0]

  pred = non_max_suppression(pred, conf_thres=0.60)[0] # conf_thres is confidence thresold

  if pred is not None:

    # scale coords to match true image size
    pred = scale_coords((h, w), img[0].shape[1:], pred)
  
  return pred
                

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

def show_detections(detections, image, ret=False, is_label=False):
  img = image.copy()
  for pred in detections:
    x1 = int(pred[0])
    y1 = int(pred[1])
    x2 = int(pred[2])
    y2 = int(pred[3])

    start = (x1,y1)
    end = (x2,y2)

    if is_label:
      pred_data = f'{label[pred[-1].item()]}'
    else:
      pred_data = f'{label[pred[-1].item()]} {str(pred[-2].item()*100)[:5]}%'

    color = (0,255,0)
    img = cv2.rectangle(img, start, end, color)
    img = cv2.putText(img, pred_data, (x1,y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA) 
  if ret: return img.copy()
  cv2.imshow('frame', img)
  cv2.waitKey(0)

def validacao(stats):

  # compute metrics
  stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
  
  if len(stats) and stats[0].any():
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats)
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(np.int64), minlength=80)  # number of targets per class
    return mp, mr, map50, map
  else:
    nt = torch.zeros(1)
    return nt


def testSingleImage():
  # load image from coco dataset
  img_path = "000000015338.jpg"
  img = cv2.imread(img_path)
  image = img.copy()

  # get labels
  labels = torch.tensor(
           [[5, 264.58,201.17,84.19,59.23],
            [2, 140.15,208.61,50.17,13.60],
            [2, 0.0,187.2,32.65,27.37],
            [5, 444.95,171.72,91.05,100.05],
            [7, 67.58,185.79,78.09,35.86]]
  )
  
  """
  Evaluate 
  """
  # labels coordinates are in coco format
  # so convert them to xyxy
  labels[:, 1:] = torch.tensor(coco2xyxy(labels[:, 1:]))

  iou = torch.linspace(0.5, 0.95, 10)

  # get detections
  detections = torch.tensor(get_pred(img))

  correct = process_batch(detections, labels, iou)
  conf = detections[:, 4]
  pred_cls = detections[:, 5]
  tcls = labels[:, 0]

  # building stats
  stats = [(correct.cpu(), conf.cpu(), pred_cls.cpu(), tcls)]
  validacao(stats)

def subjective(*stats, detections, labels, img):
  mp, mr, map50, map = validacao([stats])
  if map50 < 0.5:
    print(f"mAP50: {map50}")
    
    # get detections images and labels image with bb
    det_img = show_detections(detections, img, ret=True)
    l = torch.column_stack((labels[:, 1:], labels[:, 0]))
    lab_img = show_detections(l, img, ret=True, is_label=True)

    # concatenate det_img and lab_img vertical
    res = cv2.hconcat([det_img, lab_img])
    cv2.imshow('frame', res)
    cv2.waitKey(0)



import fiftyone as fo
import fiftyone.zoo as foz
from tqdm import tqdm

def testCOCO():
  # Download and load the validation split of COCO-2017
  dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    max_samples=100,
    shuffle=False,
  )
  dataset = list(dataset)
  stats = []
  for sample in tqdm(dataset):
    img = cv2.imread(sample.filepath)
    h,w = img.shape[:-1]

    # get labels
    labels = [[classnames[det['label']]]+det['bounding_box'] for det in sample['ground_truth']['detections']]
    labels = torch.tensor(labels)

    # denormalize coordinates
    labels[:, [1, 3]] *= w
    labels[:, [2, 4]] *= h
    # from coco 2 xyxy
    labels[:, 1:] = torch.tensor(coco2xyxy(labels[:, 1:]))

    iou = torch.linspace(0.5, 0.95, 10)
    tcls = labels[:, 0]

    # get detections
    detections = get_pred(img)
    if detections is None:
      stats.append((torch.zeros(0, iou.numel(), dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
      continue

    detections = torch.tensor(detections)

    correct = process_batch(detections, labels, iou)
    conf = detections[:, 4]
    pred_cls = detections[:, 5]

    stats_params = (correct.cpu(), conf.cpu(), pred_cls.cpu(), tcls)
    stats.append(stats_params)

    # subjective
    subjective(*stats_params, detections=detections, labels=labels, img=img)




#print("P   R   mAP50  map0.5@0.95")
if __name__ == '__main__':
  #testSingleImage()
  testCOCO()

  #show_detections(detections, img)
  #show_detections(labels[:, 1:], img)
