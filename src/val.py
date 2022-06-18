import torch
import cv2
from utils.datasets import MyDatasetMOT, MyDatasetDET
from utils.inference import Inference 
from utils.evaluator import Evaluator
from utils.general import DetectionsMatrix, non_max_suppression, Annotator
from utils.conversions import scale_coords, xywh2xyxy, xyxy2xywh
from utils.metrics import process_batch
from deep_sort.deep_sort import DeepSort
from opts import OPTS
from tqdm import tqdm
import time
import numpy as np

def run(dataset, model, opt):
  # validation
  stats = []
  annotator = Annotator()
  labels = DetectionsMatrix(model.names, model.names) # labels in mot format
  is_yolo = True if opt.model.startswith('yolo') else False
  detections = DetectionsMatrix(model.names, model.names) # detections in mot format
  iou = torch.linspace(0.5, 0.95, 10)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # load deepsort
  deepsort = DeepSort('osnet_x0_25', device, 0.2, 0.7, 30, 1, 100)
  inference_times = []
  for i,(img,targets,paths,shapes) in enumerate(tqdm(dataset, total=len(dataset))):
    #if i == 100: break
    im = cv2.imread(paths)
    img = img.to(device, non_blocking=True)

    targets = targets.to(device)
    nb, _, height, width = img.shape

    # Inference
    if is_yolo:
      out = model(img)[0]
      # NMS 
      pred = non_max_suppression(out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres)[0]
    else:
      img = cv2.imread(paths)
      # register time
      start = time.time()
      pred = model.run_on_opencv_image(img)
      inference_times.append(time.time() - start)

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

    if is_yolo:
      pred_bboxes = pred[:, :4]
      # scale detections to native coordinates
      pred_bboxes = scale_coords(img[0].shape[1:], pred_bboxes, shapes[0])
      pred_confs = pred[:, 4]
      pred_clss = pred[:, 5]
    else:
      pred_bboxes = pred.bbox
      pred_confs = pred.get_field('scores')
      pred_clss = pred.get_field('labels')-2

      indexes = pred_clss != -1
      pred_clss = pred_clss[indexes]
      pred_bboxes = pred_bboxes[indexes]
      pred_confs = pred_confs[indexes]


    # Predictions in MOT format
    detections.update_current(
      bboxes=xyxy2xywh(pred_bboxes),
      confs=pred_confs, # confs
      clss=pred_clss # clss
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
            pred_confs.cpu(),
            pred_clss.cpu(),
            labels.current[:, -1]
          )],
          detections=detections.current,
          labels=labels.current,
          img=im,
          annotator=annotator,
          classnames=model.names,
        )
      # append stats to eval detector
      if is_yolo:
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), labels.current[:, -1]))  # (correct, conf, pcls, tcls)
      else:
        stats.append((correct.cpu(), pred_confs, pred_clss, labels.current[:, -1]))
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
            classnames=model.names
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
    hotas = []
    for cls in res.keys():
      #print(cls)
      if res[cls]['HOTA'].mean() > 0:
        aux = {}
        aux['HOTA'] = res[cls]['HOTA'].mean()
        aux['DetA'] = res[cls]['DetA'].mean()
        aux['AssA'] = res[cls]['AssA'].mean()
        aux['LocA'] = res[cls]['LocA'].mean()
        hotas.append(aux)
        """
        for k in evaluator.hota.float_array_fields:
          print(k, res[cls][k].mean()*100)
        """
        """
        for k in evaluator.hota.float_fields:
          print(k, res[cls][k]*100)
        """
    return hotas
  else:
    # compute detector stats
    print(len(stats))
    results = Inference.compute_stats(stats, names=model.names)
    print("Detector: mp:{}, mr:{}, map50:{}, map:{}".format(*results))
    print(np.mean(inference_times))

  cv2.destroyAllWindows()

import glob
if __name__ == '__main__':

  opt = OPTS.val_args()
  times = []
  final_list = []
  for path in glob.glob("/home/henistein/projects/ProjetoLicenciatura/datasets/MOT/*"):
    opt.path = path

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

    print(f"Running {opt.model}")
    if opt.model.startswith('yolo'):
      # load yolo model
      model = torch.load(weights)['model'].float()
      model.to(torch.device('cuda')).eval()
    else:
      from maskrcnn_benchmark.config import cfg
      from utils.predictor import COCODemo
      # load fasterrcnn model
      config_file = "configs/fasterrcnn.yaml"
      cfg.merge_from_file(config_file)
      cfg.merge_from_list(["MODEL.WEIGHT", weights])
      model = COCODemo(
        cfg,
        min_image_size=opt.img_size,
        #min_image_size=800,
        confidence_threshold=0.7,
      )
      model.names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

    start = time.time()
    res = run(dataset, model, opt)
    times.append((time.time() - start)/len(dataset))

    #print(res)
    final = {}
    for k in ['HOTA', 'DetA', 'AssA', 'LocA']:
      final[k] = np.mean([r[k] for r in res])*100
    final_list.append(final)

  
  results = {}
  for k in ['HOTA', 'DetA', 'AssA', 'LocA']:
    results[k] = np.mean([r[k] for r in final_list])
  print(results)
  print(np.mean(times))