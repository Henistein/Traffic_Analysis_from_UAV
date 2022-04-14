import numpy as np
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
from sympy import evaluate
from utils.hota import HOTA

class Evaluator:
  def __init__(self, gt, dt, num_timesteps, valid_classes, classes_to_eval):
    self.num_timesteps = num_timesteps
    self.distractor_class_names = []
    for cls in valid_classes:
      if cls not in classes_to_eval:
        self.distractor_class_names.append(cls)
    self.distractor_class_names = []
    self.valid_classes = valid_classes
    self.class_list = classes_to_eval
    self.class_name_to_class_id = {k:i+1 for i,k in enumerate(self.valid_classes)}
    self.valid_class_numbers = list(self.class_name_to_class_id.values())
    self.gt = self._load_data(gt, is_gt=True)
    self.dt = self._load_data(dt)
    self.raw_data = self._process_data()
    self.hota = HOTA()
  
  def _check_instance(self, obj):
    if isinstance(obj, str):
      return np.loadtxt(obj, delimiter=',')
    return obj

  def _load_data(self, obj, is_gt=False):
    obj = self._check_instance(obj)

    data_keys = ['ids', 'classes', 'dets']
    if is_gt: 
      data_keys += ['gt_extras']
    else:
      data_keys += ['tracker_confidences']

    raw_data = {key: [None] * self.num_timesteps for key in data_keys}

    frame_ids = np.unique(obj[:, 0])
    
    for i in range(self.num_timesteps+1):
      if i in frame_ids:
        i = int(i) - 1
        # time indexes
        idx = obj[:, 0] == (i+1)

        raw_data['ids'][i] = np.atleast_1d(obj[idx, 1]).astype(int)
        raw_data['classes'][i] = np.atleast_1d(obj[idx, 7]).astype(int)
        raw_data['dets'][i] = np.atleast_2d(obj[idx, 2:6])
        if not is_gt: 
          raw_data['tracker_confidences'][i] = np.atleast_1d(obj[idx, 6])
        else:
          raw_data['gt_extras'][i] = {'zero_marked': np.atleast_1d(obj[idx, 6]).astype(int)}
      else:
        raw_data['dets'][i] = np.empty((0, 4))
        raw_data['ids'][i] = np.empty(0).astype(int)
        raw_data['classes'][i] = np.empty(0).astype(int)
        if is_gt:
          raw_data['gt_extras'][i] = {'zero_marked': np.empty(0)}
        else:
          raw_data['tracker_confidences'][i] = np.empty(0)

    # update dict to gt or dt
    if is_gt:
      key_map = {'ids': 'gt_ids',
                'classes': 'gt_classes',
                'dets': 'gt_dets'}
    else:
      key_map = {'ids': 'tracker_ids',
                'classes': 'tracker_classes',
                'dets': 'tracker_dets'}
    for k, v in key_map.items():
      raw_data[v] = raw_data.pop(k)
    raw_data['num_timesteps'] = self.num_timesteps

    return raw_data
  
  def _process_data(self):
    raw_data = {**self.dt, **self.gt}
    #calculate similarities for each timestep
    similarity_scores = []
    for t,(gt_dets_t, tracker_dets_t) in enumerate(zip(raw_data['gt_dets'], raw_data['tracker_dets'])):
      ious = Evaluator.calculate_box_ious(gt_dets_t, tracker_dets_t)
      similarity_scores.append(ious)
    raw_data['similarity_scores'] = similarity_scores
    return raw_data
  
  def _get_preprocessed_data(self, cls):
    data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'tracker_confidences', 'similarity_scores']
    data = {key: [None] * self.raw_data['num_timesteps'] for key in data_keys}
    unique_gt_ids = []
    unique_tracker_ids = []
    num_gt_dets = 0
    num_tracker_dets = 0
    cls_id = self.class_name_to_class_id[cls]
    distractor_classes = [self.class_name_to_class_id[x] for x in self.distractor_class_names]
    for t in range(self.raw_data['num_timesteps']):
      # Only extract revelant dets for this class preproc and eval (cls)
      gt_class_mask = np.atleast_1d(self.raw_data['gt_classes'][t] == cls_id)
      gt_class_mask = gt_class_mask.astype(bool)
      gt_classes = self.raw_data['gt_classes'][t][gt_class_mask]
      gt_ids = self.raw_data['gt_ids'][t][gt_class_mask]
      gt_dets = self.raw_data['gt_dets'][t][gt_class_mask]
      gt_zero_marked = self.raw_data['gt_extras'][t]['zero_marked'][gt_class_mask]

      tracker_class_mask = np.atleast_1d(self.raw_data['tracker_classes'][t] == cls_id)
      tracker_class_mask = tracker_class_mask.astype(bool)
      tracker_ids = self.raw_data['tracker_ids'][t][tracker_class_mask]
      tracker_dets = self.raw_data['tracker_dets'][t][tracker_class_mask]
      similarity_scores = self.raw_data['similarity_scores'][t][gt_class_mask, :][:, tracker_class_mask]
      tracker_confidences = self.raw_data['tracker_confidences'][t][tracker_class_mask]

      # Match tracker and gt dets (with hungarian algorithm) and remove tracker dets which match with gt dets
      # which are labeled as belonging to a distractor class.
      to_remove_tracker = np.array([], int)
      unmatched_indices = np.arange(tracker_ids.shape[0])
      if gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
        matching_scores = similarity_scores.copy()
        matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
        match_rows, match_cols = linear_sum_assignment(-matching_scores)
        actually_matched_mask = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps
        match_rows = match_rows[actually_matched_mask]
        match_cols = match_cols[actually_matched_mask]

        is_distractor_class = np.isin(gt_classes[match_rows], distractor_classes)
        to_remove_tracker = match_cols[is_distractor_class]
        unmatched_indices = np.delete(unmatched_indices, match_cols, axis=0)
        #to_remove_tracker = unmatched_indices


      # Apply preprocessing to remove all unwanted tracker dets.
      data['tracker_ids'][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
      data['tracker_dets'][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
      data['tracker_confidences'][t] = np.delete(tracker_confidences, to_remove_tracker, axis=0)
      similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

      # Remove gt detections marked as to remove (zero marked), and also remove gt detections not in pedestrian
      # class (not applicable for MOT15)
      gt_to_keep_mask = (np.not_equal(gt_zero_marked, 0)) & \
                        (np.equal(gt_classes, cls_id))
      data['gt_ids'][t] = gt_ids[gt_to_keep_mask]
      data['gt_dets'][t] = gt_dets[gt_to_keep_mask, :]
      data['similarity_scores'][t] = similarity_scores[gt_to_keep_mask]

      unique_gt_ids += list(np.unique(data['gt_ids'][t]))
      unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
      num_tracker_dets += len(data['tracker_ids'][t])
      num_gt_dets += len(data['gt_ids'][t])
    # Re-label IDs such that there are no empty IDs
    if len(unique_gt_ids) > 0:
      unique_gt_ids = np.unique(unique_gt_ids)
      gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
      gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
      for t in range(self.raw_data['num_timesteps']):
        if len(data['gt_ids'][t]) > 0:
          data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(int)
    
    if len(unique_tracker_ids) > 0:
      unique_tracker_ids = np.unique(unique_tracker_ids)
      tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
      tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
      for t in range(self.raw_data['num_timesteps']):
        if len(data['tracker_ids'][t]) > 0:
          data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(int)

    # Record overview statistics.
    data['num_tracker_dets'] = num_tracker_dets
    data['num_gt_dets'] = num_gt_dets
    data['num_tracker_ids'] = len(unique_tracker_ids)
    data['num_gt_ids'] = len(unique_gt_ids)
    data['num_timesteps'] = self.raw_data['num_timesteps']

    return data


  @staticmethod
  def calculate_box_ious(gt_dets, tracker_dets):
    # layout: (x0, y0, w, h)
    bboxes1 = deepcopy(gt_dets)
    bboxes2 = deepcopy(tracker_dets)

    bboxes1[:, 2] = bboxes1[:, 0] + bboxes1[:, 2]
    bboxes1[:, 3] = bboxes1[:, 1] + bboxes1[:, 3]
    bboxes2[:, 2] = bboxes2[:, 0] + bboxes2[:, 2]
    bboxes2[:, 3] = bboxes2[:, 1] + bboxes2[:, 3]

    # layout: (x0, y0, x1, y1)
    min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    intersection = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])

    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection
    intersection[area1 <= 0 + np.finfo('float').eps, :] = 0
    intersection[:, area2 <= 0 + np.finfo('float').eps] = 0
    intersection[union <= 0 + np.finfo('float').eps] = 0
    union[union <= 0 + np.finfo('float').eps] = 1
    ious = intersection / union

    return ious
  
  def run_hota(self):
    # run hota metrics
    res = {} 
    for cls in self.class_list:
      data = self._get_preprocessed_data(cls)
      res[cls] = self.hota.eval_sequence(data)
    return res 

if __name__ == '__main__':
  float_array_fields = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'RHOTA']
  float_fields = ['HOTA(0)', 'LocA(0)', 'HOTALocA(0)']
  evaluator = Evaluator(
    gt='gt.txt',
    dt='visdrone.txt',
    num_timesteps=500,
    valid_classes= ['pedestrian', 'bicycle', 'people', 'car', 'van', 'truck', 'trycycle', 'awning-tricycle', 'bus', 'motor'],
    classes_to_eval=['car', 'truck']
  )

  res = evaluator.run_hota()
  for cls in res.keys():
    print(cls)
    for k in float_array_fields:
      print(k, res[cls][k].mean()*100)
    for k in float_fields:
      print(k, res[cls][k]*100)