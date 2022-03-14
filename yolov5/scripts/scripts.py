import glob
import shutil
from pathlib import Path
from tqdm import tqdm

def filter_visdrone_classes(imgs_path, labels_path, out_path):
  images = glob.glob(imgs_path + "/*")
  names = [Path(f).stem for f in images]
  classes = [3, 4, 5, 6, 7, 8, 9, 10]

  out_img_path = out_path + "/images"
  out_label_path = out_path + "/annotations"

  for name in tqdm(names):
    img_file = glob.glob(imgs_path+f"/{name}.*")
    label_file = glob.glob(labels_path+f"/{name}.*")

    f_aux = open(label_file[0])
    new_label_data = ""
    for line in f_aux.readlines():
      data = line.split(',')
      bbox = data[:4]
      cls = data[-3]

      # filter class
      cls = int(data[-3])
      if cls not in classes:
        continue
      cls = str(cls)

      # add lines to new label data
      new_label_data += (",".join([cls]+bbox) + '\n')

    if len(new_label_data):
      # remove last \n
      new_label_data = new_label_data[:-1]
      # write data
      f_out_label = open(out_label_path+f"/{name}.txt", 'w')
      f_out_label.write(new_label_data)
      # move img to new location
      shutil.copy(img_file[0], out_img_path)

filter_visdrone_classes(
    "/home/henistein/projects/ProjetoLicenciatura/datasets/VisDrone/VisDroneVal/images",
    "/home/henistein/projects/ProjetoLicenciatura/datasets/VisDrone/VisDroneVal/annotations",
    "/home/henistein/projects/ProjetoLicenciatura/datasets/VisDrone/VisDroneValFiltered"
)

