import torch
import unittest
import fiftyone as fo
import fiftyone.zoo as foz
import cv2
from tqdm import tqdm
from utils.conversions import coco2xyxy
from inference import get_pred, compute_metrics
from utils.metrics import process_batch

class Test(unittest.TestCase):
  def setUp(self):
    classnames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    self.classnames = {k:classnames.index(k) for k in classnames}
    self.weights = 'weights/yolov5x.pt'
    self.model = torch.load(self.weights)['model'].float()
    self.model.eval()

  def test_coco_evaluation(self):
    """
    Test on coco validation datast
    """
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
      labels = [[self.classnames[det['label']]]+det['bounding_box'] for det in sample['ground_truth']['detections']]
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

    # compute stats
    mp, mr, map50, mapp = compute_metrics(stats)

    self.assertTrue(mp >= 0.80)
    self.assertTrue(mr >= 0.58)
    self.assertTrue(map50 >= 0.70)
    self.assertTrue(mapp >= 0.54)


if __name__ == '__main__':
    unittest.main()
