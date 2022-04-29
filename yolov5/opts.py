import argparse

class OPTS:
  @staticmethod 
  def val_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='', help='path to dataset')
    parser.add_argument('--model', type=str, default='yolov5l.pt', help='model (yolov5l.pt or yolov5l-xs.pt')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='NMS confident threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS iou threshold')
    parser.add_argument('--img-size', type=int, default=640, help='inference img size')
    parser.add_argument('--subjective', action='store_true', help='show two frames, one with predictions and other with gt labels ')
    parser.add_argument('--detector', action='store_true', help='evaluate detector performance (mAP)')
    parser.add_argument('--strongsort', action='store_true', help='run strongsort tracker')

    return parser.parse_args()
  
  @staticmethod
  def main_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='', help='path to dataset')
    parser.add_argument('--model', type=str, default='yolov5l-xs.pt', help='model (yolov5l.pt or yolov5l-xs.pt')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='NMS confident threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS iou threshold')
    parser.add_argument('--img-size', type=int, default=640, help='inference img size')
    parser.add_argument('--start-from', type=int, default=0, help='video start miliseconds')
    parser.add_argument('--video-out', action='store_true', help='outputs the video')
    parser.add_argument('--labels-out', action='store_true', help='outputs the labels')
    parser.add_argument('--no-show', action='store_true', help='do not show the video')
    parser.add_argument('--just-detector', action='store_true', help='just run the detector')

    return parser.parse_args()