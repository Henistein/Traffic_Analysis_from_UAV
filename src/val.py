import torch
import cv2
from datasets import MyDatasetMOT, MyDatasetDET
from inference import Inference 
from utils.evaluator import Evaluator
from utils.general import DetectionsMatrix, non_max_suppression, Annotator
from utils.conversions import scale_coords, xywh2xyxy, xyxy2xywh
from utils.metrics import process_batch
from deep_sort.deep_sort import DeepSort
from opts import OPTS
from tqdm import tqdm
import numpy as np

def run(dataset, model, opt, device):
  # validation
  model.eval()
  classnames = model.names
  stats = []
  annotator = Annotator()
  labels = DetectionsMatrix(model.names, model.names) # labels in mot format
  detections = DetectionsMatrix(model.names, model.names) # detections in mot format
  iou = torch.linspace(0.5, 0.95, 10)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # load deepsort
  deepsort = DeepSort('osnet_x0_25', device, 0.2, 0.7, 30, 1, 100)

  print('Tracker: '+'StrongSort' if opt.strongsort else 'DeepSort')

  for i,(img,targets,paths,shapes) in enumerate(tqdm(dataset, total=len(dataset))):
    im = cv2.imread(paths)
    img = img.to(device, non_blocking=True)

    targets = targets.to(device)
    nb, _, height, width = img.shape

    # Inference
    out = model(img)[0]

    # NMS 
    pred = non_max_suppression(out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres)[0]
    if pred is None or len(pred) == 0:
      if opt.detector:
        stats.append((torch.zeros(0, iou.numel(), dtype=torch.bool), torch.Tensor(), torch.Tensor(), targets[:, -1].cpu().numpy()))
      continue 

    # labels in MOT format
    if not opt.detector:
      labels.update_current(
        ids=targets[:, 0],
        bboxes=targets[:, 1:5],
        confs=None,
        clss=targets[:, 5]
      )
    else:
      labels.update_current(
        ids=None,
        bboxes=targets[:, :4],
        confs=None,
        clss=targets[:, 4]
      )

    # scale detections to native coordinates
    pred[:, 0:4] = scale_coords(img[0].shape[1:], pred[:, 0:4], shapes[0])

    # Predictions in MOT format
    detections.update_current(
      bboxes=xyxy2xywh(pred[:, 0:4]),
      confs=pred[:, 4], # confs
      clss=pred[:, 5] # clss
    )

    # pass detections to deepsort
    if not opt.detector:
      outputs = deepsort.update(
        torch.tensor(detections.current[:, 2:6]), # xywhs
        torch.tensor(detections.current[:, 6]), # confs
        torch.tensor(detections.current[:, 7]).int(), # clss
        im.copy()
      )

    # Evaluate detector (mAP)
    if opt.detector:
      # convert from xywh to xyxy
      detections.current[:, 2:] = xywh2xyxy(detections.current[:, 2:])
      correct = process_batch(detections.current[:, 2:], labels.current[:, 2:], iou)
      # visualize
      if opt.subjective:
        # Compute 2 imgs, one with gt labels and other with detections labels
        Inference.subjective(
          stats=[(
            correct.cpu(),
            pred[:, 4].cpu(),
            pred[:, 5].cpu(),
            labels.current[:, -1]
          )],
          detections=detections.current,
          labels=labels.current,
          img=im,
          annotator=annotator,
          classnames=classnames
        )
      # append stats to eval detector
      stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), labels.current[:, -1]))  # (correct, conf, pcls, tcls)
    else: 
      if len(outputs) > 0:
        min_dim = min(outputs.shape[0], detections.current.shape[0])
        outputs = outputs[:min_dim]
        detections.current = detections.current[:min_dim]
        detections.current[:, 2:6] = outputs[:, :4] # bboxes
        detections.current[:, 1] = outputs[:, 4] + 1 # ids
        detections.current[:, 7] = outputs[:, -2]
      
        # visualize
        if opt.subjective:
          # Compute 2 imgs, one with gt labels and other with detections labels
          Inference.subjective(
            detections=detections.current,
            labels=labels.current,
            img=im,
            annotator=annotator,
            classnames=classnames
          )

    # update mot_matrix
    detections.update()
    labels.update()

  if not opt.detector:
    # remove -1
    detections.mot_matrix = detections.mot_matrix[detections.mot_matrix[:, 1] != -1]
    # compute tracking stats
    evaluator = Evaluator(
      gt=labels.mot_matrix,
      dt=detections.mot_matrix,
      num_timesteps=len(dataset),
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

  else:
    # compute detector stats
    results = Inference.compute_stats(stats)
    print("Detector: mp:{}, mr:{}, map50:{}, map:{}".format(*results))

  cv2.destroyAllWindows()

if __name__ == '__main__':

  opt = OPTS.val_args()

  # dataset path
  if not opt.detector:
    dataset = MyDatasetMOT(
        imgs_path=opt.path + '/images',
        labels_path=opt.path + '/labels.txt',
        imsize=opt.img_size
    )
  else:
    dataset = MyDatasetDET(
        imgs_path=opt.path + '/images',
        labels_path=opt.path + '/annotations',
        imsize=opt.img_size
    )

  # model
  weights = 'weights/' + opt.model
  device = torch.device('cuda')
  model = torch.load(weights)['model'].float()
  model.to(device)

  run(dataset, model, opt, device)
