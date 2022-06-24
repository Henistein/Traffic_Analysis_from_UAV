#!/bin/bash
python3 val.py --path /home/henistein/projects/ProjetoLicenciatura/datasets/MOT/test_dev10 --model yolov5l-xs.pt  --img-size 1920

#python3 val.py --path /home/henistein/projects/ProjetoLicenciatura/datasets/VisDrone2019-DET-val --model visdrone_model_0360000.pth --detector --conf-thres 0.001 --iou-thres 0.60 --img-size 1920 #--subjective
#python3 val.py --path /home/henistein/projects/ProjetoLicenciatura/datasets/VisDrone2019-DET-val --model yolov5l-xs.pt --detector --conf-thres 0.7 --iou-thres 0.50 --img-size 640 #--subjective

#python3 val.py --path /home/henistein/projects/ProjetoLicenciatura/datasets/test_dev1 --model yolov5l-xs.pt #--subjective
#python3 val.py --path /home/henistein/projects/ProjetoLicenciatura/datasets/test_dev1 --model visdrone_model_0360000.pth --subjective
