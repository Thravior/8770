import numpy as np
import math
from numpy.linalg import norm
from Katna.video import Video
import Katna.writer as kw
import cv2
from multiprocessing import Process, Queue
import csv
import torch
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
from einops import rearrange
from IPython.display import display
import matplotlib.pyplot as plt

def CalcSimCos(base:np.ndarray, ref:np.ndarray):
  local_b = base.flatten()
  local_r = ref.flatten()
  return (np.dot(local_b,local_r)/(norm(local_r)*norm(local_b)))

def CalcDist(base, ref:np.ndarray):
  local_b = base.flatten()/base.sum()
  local_r = ref.flatten()/ref.sum()
  dist:np.ndarray = np.absolute([local_b - local_r])
  return dist.sum()

def CalculHisto3D(image: np.ndarray): ## Taille 64
  image.astype('uint8')
  hist:np.ndarray = cv2.calcHist([image], [0,1,2], None, [4,4,4], [0, 256,0, 256,0, 256],).astype("uint32")
  image -= 32
  hist2 = cv2.calcHist([image], [0,1,2], None, [4,4,4], [0, 256,0, 256,0, 256]).astype("uint32")
  concatenated:np.ndarray =  np.concatenate([hist,hist2])
  t = np.array([concatenated & 0x0000FFFF,concatenated & 0x00FF00FF,concatenated & 0x00FFFF00],np.uint8).flatten()
  return concatenated

def extract_frames1s(video_path, couleur):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return None

    frames = []
    frame_count = 0
    # Read and process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to numpy array and store it
        frame_np =couleur(np.asarray(frame)) 
        frames.append((CalculHisto3D(frame_np),frame_count))

        frame_count += 1

        # Skip frames to keep 3 frame per second
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * cap.get(cv2.CAP_PROP_FPS))

    cap.release()

    return (video_path.split('\\')[-1][:4],frames)

def extract_frames3ps(video_path, couleur):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return None

    frames = []
    frame_count = 0
    # Read and process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to numpy array and store it
        frame_np =couleur(np.asarray(frame)) 
        frames.append((CalculHisto3D(frame_np),frame_count/3))

        frame_count += 1

        # Skip frames to keep 3 frame per second
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * cap.get(cv2.CAP_PROP_FPS)//3)

    cap.release()

    return (video_path.split('\\')[-1][:4],frames)

def GetBestCorr(liste):
  best = liste[0]
  for i in range(1,len(liste)):
    if liste[i][1] < best[1]:
        best = liste[i]
  return best

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1.transforms)   # le modèle est chargé avec des poids pré-entrainés sur ImageNet
model = torch.nn.Sequential(*(list(model.children())[:-1]))        # supprime la dernière couche du réseau
model.eval();   

def Encode_image(image):
   with torch.no_grad():
    output = model(image)
    print("output encode", output)  # 1 x 512 x 1 x 1 


if __name__ == '__main__':
  import time
  now = time.time()

  # Espace de couleur avec leur seuil
  fonctions_color = [lambda x: cv2.cvtColor(x,cv2.COLOR_BGR2YUV), lambda x: x]
  seuils = [0.6,0.42]
  desc_color = ["YUV", "RGB"]
  # Exctraction des videos
  fonctions_temps = [extract_frames3ps,extract_frames1s]
  desc_temps = ["3ps","1ps"]
  for c in range(len(fonctions_color)):
    f_color = fonctions_color[c]

    for f in range(len(fonctions_temps)):
      f_extract = fonctions_temps[f]

      # 
      if c == 0 and f == 0:
         continue

      captures = []
      
      now = time.time()

      for target in range(1,101):
        #video_path = "TP3\moodle\data\mp4\\v" + str(target).zfill(3) +  ".mp4"
        video_path = "TP3/moodle/data/mp4/v001.mp4"
        captures.append(f_extract(video_path,f_color))

      temps_indexation = time.time() - now

      print("Durée indexation:")
      print(temps_indexation)

      csv_file_path = "results_"+desc_temps[f]+"s_ReseauxNeurones.csv"

      now = time.time()
      with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(("image","video_pred","minutage_pred"))

        for i in range(0,1000):
          #image_path = "TP3\moodle\data\jpeg\i" + str(i).zfill(3) +".jpeg"
          #home/marie-claire/Documents/8770/8770/TP3/moodle/data/jpeg
          image_path = "TP3/moodle/data/jpeg/i000.jpeg"
          image = Image.open(image_path)
          preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                  std=[0.229,0.224,0.225])
          ])
          input_tensor = preprocess(image)
          input_image = input_tensor.unsqueeze(0)
          histogram = Encode_image(input_image)
          histogram:np.ndarray
          res = []
          for video in captures:
            rows_b = []
            rows_m = []
            for frame in video[1]:
              dist = CalcDist(histogram,frame[0])
              if dist < seuils[c]:
                rows_b.append((frame[1],dist))

            if len(rows_b) > 0:
              b = GetBestCorr(rows_b)
              res.append((video[0],b[0],b[1]))
  
          final = None 
          if len(res) == 0:
            final = ("out","",1)
          else: 
            res.sort(key= lambda x: x[2])  
            final = res[0]

          csv_writer.writerow(("i"+str(i).zfill(3), final[0],final[1]))
        csv_writer.writerow(("indexation", temps_indexation,"Tests",time.time() - now))
