import torch
import cv2
from datasets import MyDataset
from inference import Inference, Annotator
from utils.evaluator import Evaluator
from utils.general import DetectionsMatrix, non_max_suppression
from utils.conversions import scale_coords, xywh2xyxy, xyxy2xywh
from utils.metrics import process_batch
from deep_sort.deep_sort import DeepSort
from opts import OPTS
from tqdm import tqdm

def run(dataset, model, opt, device):
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
  deepsort = DeepSort('osnet_ibn_x1_0_MSMT17', device, strongsort=opt.strongsort)
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
        stats.append((torch.zeros(0, iou.numel(), dtype=torch.bool), torch.Tensor(), torch.Tensor(), labels.current[:, -1]))
      continue 

    # labels in MOT format
    labels.update_current(
      ids=targets[:, 0],
      bboxes=targets[:, 1:5],
      confs=None,
      clss=targets[:, 5]
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
    outputs = deepsort.update(
      detections.current[:, 2:6], # xywhs
      detections.current[:, 6], # confs
      detections.current[:, 7], # clss
      im.copy()
    )

    # Evaluate detector (mAP)
    if opt.detector:
      correct = process_batch(xywh2xyxy(detections.current[:, 2:]), labels.current[:, 2:], iou)
    
    if len(outputs) > 0:
      min_dim = min(outputs.shape[0], detections.current.shape[0])
      outputs = outputs[:min_dim]
      detections.current = detections.current[:min_dim]
      detections.current[:, 2:6] = outputs[:, :4] # bboxes
      detections.current[:, 1] = outputs[:, 4] + 1 # ids
      detections.current[:, 7] = outputs[:, -1]

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
    if opt.detector:
      stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), labels.current[:, -1]))  # (correct, conf, pcls, tcls)

    # update mot_matrix
    detections.update()
    labels.update()

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

  # compute detector stats
  if opt.detector:
    results = Inference.compute_stats(stats)
    print(results)
  cv2.destroyAllWindows()

if __name__ == '__main__':

  opt = OPTS.val_args()

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

  run(dataset, model, opt, device)