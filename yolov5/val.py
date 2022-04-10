import torch
import cv2
import argparse
from datasets import MyDataset
from inference import Inference, Annotator
from utils.general import non_max_suppression
from utils.conversions import scale_coords, xywh2xyxy
from utils.metrics import process_batch
from tqdm import tqdm

def parse_opt():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', type=str, default='', help='path to dataset')
  parser.add_argument('--model', type=str, default='yolov5l.pt', help='model (yolov5l.pt or yolov5l-xs.pt')
  parser.add_argument('--conf-thres', type=float, default='0.5', help='NMS confident threshold')
  parser.add_argument('--iou-thres', type=float, default='0.5', help='NMS iou threshold')
  parser.add_argument('--img-size', type=int, default='640', help='inference img size')
  parser.add_argument('--subjective', action='store_true', help='show two frames, one with predictions and other with gt labels ')

  return parser.parse_args()

def run(dataset, model, conf_thres, iou_thres, subjective, device):
  # validation
  model.eval()
  classnames = model.names
  stats = []
  annotator = Annotator()
  iou = torch.linspace(0.5, 0.95, 10)

  for i,(img,targets,paths,shapes) in enumerate(tqdm(dataset, total=len(dataset))):
    img = img.to(device, non_blocking=True)
    img = img.float() / 255

    targets = targets.to(device)
    nb, _, height, width = img.shape

    # Inference
    out = model(img)[0]
    # NMS 
    targets[:, 1:5] *= torch.tensor((width, height, width, height), device=device) # to pixels
    out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres)
    # scale bbox to native coordinates
    bbox = targets[:, 1:5].clone()
    nl = len(bbox)
    shape = shapes[0]
    tbox = xywh2xyxy(bbox).cpu() # target boxes 
    scale_coords(img[0].shape[1:], tbox, shape, shapes[1])  # native-space labels
    # labels in [cls bbox] format
    tcls = targets[:, -1].tolist() if nl else []
    cls_torch = torch.tensor(tcls).reshape(-1, 1)
    labelsn = torch.cat((cls_torch, tbox), 1)
    # labels in [bbox conf cls] format
    lcc = Inference.labels_conf_cls(labels=labelsn[:, 1:], conf=None, cls=labelsn[:, 0]) # labels conf cls format

    # Metrics
    for si, pred in enumerate(out):
      if pred is None or len(pred) == 0:
        if nl:
          stats.append((torch.zeros(0, iou.numel(), dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
          continue 
      
      # Predictions
      predn = pred.clone()
      scale_coords(img[0].shape[1:], predn[:, :4], shape, shapes[1]) # natice-space pred

      # Evaluate
      if nl:
        correct = process_batch(predn.cpu(), labelsn, iou)
        # Filter just the objects on the road
        #predn[:, :4] = Inference.filter_objects_on_road(predn[:, :4], road_area)
        #tbox = Inference.filter_objects_on_road(tbox, road_area)

      # visualize
      if subjective:
        im = cv2.imread(paths)
        # Compute 2 imgs, one with gt labels and other with detections labels
        Inference.subjective(
          stats=[(
            correct.cpu(),
            pred[:, 4].cpu(),
            pred[:, 5].cpu(),
            tcls
          )],
          detections=predn,
          labels=lcc,
          img=im,
          annotator=annotator,
          classnames=classnames
        )

      stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)


  # compute stats
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

  run(dataset, model, opt.conf_thres, opt.iou_thres, opt.subjective, device)


