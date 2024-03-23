import numpy as np
from numpy.linalg import norm
import math
import cv2


def CalcSimCos(base:np.ndarray, ref:np.ndarray):
  local_b = base.flatten()
  local_r = ref.flatten()
  return (np.dot(local_b,local_r)/norm(local_r))


# Paramêtres:
TailleBinIntensite = 8
def CalculHHisto(image: np.array):
  image.astype('uint8')

  #image.divide(TailleBinIntensite)

  hist = cv2.calcHist([image], [0, 1, 2], None, [256//TailleBinIntensite,256//TailleBinIntensite,256//TailleBinIntensite], [0, 256, 0, 256, 0, 256])
  return hist/hist.sum()

image_path = "TP3\moodle\data\jpeg\i000.jpeg"
image = cv2.imread(image_path)

histogram1 = CalculHHisto(image=image)
print(type(histogram1))

image_path = "TP3\moodle\data\jpeg\i000.jpeg"
image = cv2.imread(image_path)
histogram2 = CalculHHisto(image=image)

import time

v1 = CalcSimCos(histogram1,histogram2)
