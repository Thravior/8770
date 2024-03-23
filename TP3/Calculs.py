import numpy as np
from numpy.linalg import norm
from Katna.video import Video
import Katna.writer as kw
import cv2


def CalcSimCos(base:np.ndarray, ref:np.ndarray):
  local_b = base.flatten()
  local_r = ref.flatten()
  return (np.dot(local_b,local_r)/norm(local_r))


# Paramêtres:
TailleBinIntensite = 8
def CalculHisto(image: np.array):
  image.astype('uint8')

  hist = cv2.calcHist([image], [0, 1, 2], None, [256//TailleBinIntensite,256//TailleBinIntensite,256//TailleBinIntensite], [0, 256, 0, 256, 0, 256])
  return hist/hist.sum()


class DataWriter(kw.Writer):
  """Custom writer to print the data
  """
  DataList = []
    

  def write(self, filepath, data):
    """The write method to process data generated by Katna Library
    """
    Histo = []
    for image in data:
      Histo.append(CalculHisto(image))
    # Retirer histogrammes trop similaire ???
    DataWriter.DataList.append((filepath.split('\\')[-1],Histo))
  
if __name__ == '__main__':
  image_path = "TP3\moodle\data\jpeg\i000.jpeg"
  video_path = "TP3\moodle\data\mp4\\v003.mp4"
  Vd = Video()
  writer = DataWriter()
  Vd.extract_video_keyframes(1,video_path,writer)
