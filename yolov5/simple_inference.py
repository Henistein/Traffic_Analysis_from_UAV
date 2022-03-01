import torch
import cv2
import numpy as np
from utils import non_max_suppression
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

def yolobbox2bbox(x,y,w,h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1, y1, x2, y2


def get_pred(img):
    '''
    returns prediction in numpy array
    '''
    imsize = 640
    img, h, w = image_loader(img,imsize)
    pred = model(img)[0]

    print(pred.shape)
    print(pred)
    pred = non_max_suppression(pred, conf_thres=0.40) # conf_thres is confidence thresold
    print(pred[0])

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

image = cv2.imread(path)

if image is not None:
    prediction = get_pred(image)
    print(prediction)

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
            image = cv2.putText(image, pred_data, (x1,y1+25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA) 
        cv2.imwrite(opt.output_dir+'result.jpg', image)

else:
    print('[ERROR] check input image')
    




