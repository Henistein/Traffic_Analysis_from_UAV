import torch
import cv2
import argparse
import numpy as np
from datasets import MyDataset
from inference import Inference, Annotator
from utils.evaluator import Evaluator
from utils.general import DetectionsMatrix, non_max_suppression
from utils.conversions import scale_coords, xywh2xyxy, xyxy2xywh
from utils.metrics import process_batch
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
from tqdm import tqdm

def parse_opt():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', type=str, default='', help='path to dataset')
  parser.add_argument('--model', type=str, default='yolov5l.pt', help='model (yolov5l.pt or yolov5l-xs.pt')
  parser.add_argument('--conf-thres', type=float, default='0.5', help='NMS confident threshold')
  parser.add_argument('--iou-thres', type=float, default='0.5', help='NMS iou threshold')
  parser.add_argument('--img-size', type=int, default='640', help='inference img size')
  parser.add_argument('--subjective', action='store_true', help='show two frames, one with predictions and other with gt labels ')
  parser.add_argument('--detector', action='store_true', help='evaluate detector performance (mAP)')

  return parser.parse_args()

def run(dataset, model, conf_thres, iou_thres, subjective, device, detector=False):
  # validation
  model.eval()
  classnames = model.names
  stats = []
  annotator = Annotator()
  labels = DetectionsMatrix() # labels in mot format
  detections = DetectionsMatrix() # detections in mot format
  iou = torch.linspace(0.5, 0.95, 10)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # load deepsort
  cfg = get_config()
  cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
  deepsort = DeepSort('osnet_ibn_x1_0_MSMT17',
                      device,
                      max_dist=cfg.DEEPSORT.MAX_DIST,
                      max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                      max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
  )

  for i,(img,targets,paths,shapes) in enumerate(tqdm(dataset, total=len(dataset))):
    if i == 10: break
    im = cv2.imread(paths)
    img = img.to(device, non_blocking=True)
    img = img.float() / 255

    targets = targets.to(device)
    nb, _, height, width = img.shape

    # Inference
    out = model(img)[0]
    # NMS 
    out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres)
    targets[:, 1:5] *= torch.tensor((width, height, width, height), device=device) # to pixels

    # labels in MOT format
    labels.update_current(
      ids=targets[:, 0],
      bboxes=targets[:, 1:5],
      confs=None,
      clss=targets[:, 5]
    )

    nl = len(labels.current[:, 2:6])
    # scale bbox to native coordinates, it will be stored in self.scaled_current
    labels.scale_to_native(img[0].shape[1:], shapes[0], shapes[1])

    # Metrics
    for si, pred in enumerate(out):
      if pred is None or len(pred) == 0:
        if nl and detector:
          stats.append((torch.zeros(0, iou.numel(), dtype=torch.bool), torch.Tensor(), torch.Tensor(), labels.current[:, -1]))
        continue 
      
      # Predictions in MOT format
      detections.update_current(
        bboxes=xyxy2xywh(pred[:, 0:4]),
        confs=pred[:, 4], # confs
        clss=pred[:, 5] # clss
      )

      # pass detections to deepsort
      outputs = deepsort.update(
        detections.current[:, 2:6], # xywhs
        detections.current[:, 6], # confs
        detections.current[:, 7], # clss
        im.copy()
      )

      # Evaluate detector (mAP)
      if nl and detector:
        # scale bbox to native coordinates, it will be stored in self.scaled_current
        detections.scale_to_native(img[0].shape[1:], shapes[0], shapes[1])
        correct = process_batch(detections.scaled_current[:, 2:], labels.scaled_current[:, 2:], iou)
      
      if len(outputs) > 0:
        min_dim = min(outputs.shape[0], detections.current.shape[0])
        outputs = outputs[:min_dim]
        detections.current = detections.current[:min_dim]
        detections.current[:, 2:6] = xyxy2xywh(outputs[:, :4]) # bboxes
        detections.current[:, 1] = outputs[:, 4] + 1 # ids
        detections.scale_to_native(img[0].shape[1:], shapes[0], shapes[1])
        detections.update_mot_matrix()


        # visualize
        if subjective:
          # Compute 2 imgs, one with gt labels and other with detections labels
          Inference.subjective(
            stats=[(
              correct.cpu(),
              pred[:, 4].cpu(),
              pred[:, 5].cpu(),
              labels.current[:, -1]
            )],
            detections=detections.scaled_current,
            labels=labels.scaled_current,
            img=im,
            annotator=annotator,
            classnames=classnames
          )

      # append stats to eval detector
      if detector:
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), labels.current[:, -1]))  # (correct, conf, pcls, tcls)
      # update mot_matrix with current label
      labels.update_mot_matrix()

  evaluator = Evaluator(
    gt=labels.mot_matrix,
    dt=detections.mot_matrix,
    num_timesteps=10,
    valid_classes=model.names,
    classes_to_eval=model.names
  )
  res = evaluator.run_hota()
  for cls in res.keys():
    print(cls)
    for k in evaluator.hota.float_array_fields:
      print(k, res[cls][k].mean()*100)
    for k in evaluator.hota.float_fields:
      print(k, res[cls][k]*100)

  # compute detector stats
  if detector:
    results = Inference.compute_stats(stats)
    print(results)
  cv2.destroyAllWindows()

if __name__ == '__main__':
  opt = parse_opt()

  # dataset path
  dataset = MyDataset(
      imgs_path=opt.path + '/images',
      labels_path=opt.path + '/labels.txt',
      imsize=opt.img_size
  )
  # model
  weights = 'weights/' + opt.model
  device = torch.device('cuda')
  model = torch.load(weights)['model'].float()
  model.to(device)

  run(dataset, model, opt.conf_thres, opt.iou_thres, opt.subjective, device, opt.detector)

