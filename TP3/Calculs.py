import numpy as np
import math
from numpy.linalg import norm
from Katna.video import Video
import Katna.writer as kw
import cv2
from multiprocessing import Process, Queue
import csv

def CalcSimCos(base:np.ndarray, ref:np.ndarray):
  local_b = base.flatten()
  local_r = ref.flatten()
  return 1 - (np.dot(local_b,local_r)/(norm(local_r)*norm(local_b)))

def CalcDist(base:np.ndarray, ref:np.ndarray):
  local_b = base.flatten()/base.sum()
  local_r = ref.flatten()/ref.sum()
  dist:np.ndarray = np.absolute([local_b - local_r])
  return dist.sum()

def CalculHisto(image: np.ndarray,taille):
  image.astype('uint8')
  flat = image.flatten()
  hist = cv2.calcHist([flat], [0], None, [256//taille], [0, 256])
  image -= taille//2
  hist2 = cv2.calcHist([flat], [0], None, [256//taille], [0, 256])
  return [hist,hist2]

def CalculHisto3D(image: np.ndarray, taille):
  image.astype('uint8')
  hist = cv2.calcHist([image], [0,1,2], None, [256//taille,256//taille,256//taille], [0, 256,0, 256,0, 256])
  image -= taille//2
  hist2 = cv2.calcHist([image], [0,1,2], None, [256//taille,256//taille,256//taille], [0, 256,0, 256,0, 256])
  return [hist,hist2]

def extract_frames(video_path,histogramCalculation, taille):
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
        frame_np =cv2.cvtColor(np.asarray(frame),cv2.COLOR_BGR2YUV) 
        frames.append((histogramCalculation(frame_np,taille),frame_count/3))

        frame_count += 1

        # Skip frames to keep 3 frame per second
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * cap.get(cv2.CAP_PROP_FPS)//3)

    cap.release()

    return (video_path.split('\\')[-1][:4],frames)

if __name__ == '__main__':
  import time
  now = time.time()

  fonctions_histo = [lambda x,y: CalculHisto(x,y)[0],lambda x,y: np.concatenate(CalculHisto(x,y)), lambda x,y: CalculHisto3D(x,y)[0],lambda x,y: np.concatenate(CalculHisto3D(x,y))]
  fonctions_dist = [CalcSimCos, CalcDist]

  for h in range(len(fonctions_histo)):
    f_hist = fonctions_histo[h]

    for d in range(len(fonctions_dist)):
      f_dist = fonctions_dist[d]

      for t in range(5,7):
        captures = []
        taille = 2**t
        now = time.time()

        for target in range(1,101):
          video_path = "TP3\moodle\data\mp4\\v" + str(target).zfill(3) +  ".mp4"
          captures.append(extract_frames(video_path,f_hist,taille))

        temps_indexation = time.time() - now

        print("Durée indexation:")
        print(temps_indexation)

        def GetBestCorr(liste):
          best = liste[0]
          for i in range(1,len(liste)):
            if liste[i][1] < best[1]:
                best = liste[i]
          return best

        csv_file_path = "results_histYUV"+ str(h)+ "_taille" + str(t) + "_dist" + str(d) + ".csv"

        now = time.time()
        with open(csv_file_path, 'w', newline='') as csvfile:
          with open("TP3\moodle\data\gt.csv", 'r', newline='') as TruthFile:
            csv_writer = csv.writer(csvfile)
            headers = TruthFile.readline()
            bad = 0
            far = 0
            for i in range(0,1000):
              image, reponse, minutage = TruthFile.readline().split(',')

              image_path = "TP3\moodle\data\jpeg\i" + str(i).zfill(3) +".jpeg"
              histogram = f_hist(cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2YUV),taille)
              res = []
              for video in captures:
                rows_b = []
                rows_m = []
                for frame in video[1]:
                  dist = f_dist(histogram,frame[0])
                  if dist < 0.35:
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
              if final[0] != reponse:
                csv_writer.writerow(("i"+str(i).zfill(3), final[0],final[1],final[2], "Bad" + reponse))
                bad += 1
              elif final[2] > 0.25 and final[0] != "out":
                csv_writer.writerow(("i"+str(i).zfill(3), final[0],final[1],final[2], "Far"))
                far += 1

            csv_writer.writerow(("Bad", bad,"Far",far))
          csv_writer.writerow(("indexation", temps_indexation,"Tests",time.time() - now))
