from PIL import Image
import cv2
import numpy as np

with Image.open(r"TP2\\data\\kodim01.png") as im:
    red = list(im.getdata(0))
    green = list(im.getdata(1))
    blue = list(im.getdata(2))
    array = np.asarray(im)    




    """
      Y = (R + 2G + B) /4
      U = B − G
      V = R − G 
    """
def ToYUV(data:np.array ):
    b = np.empty_like(a)
    for i in range(data.shape[0]):
       for j in range(data.shape[1]):
         b[i][j][0] = (data[i][j][0] + 2*data[i][j][1] + data[i][j][2])/4          
         b[i][j][1] = data[i][j][2] - data[i][j][1]
         b[i][j][2] = data[i][j][0] - data[i][j][1]
    return b

    """
      R = V + G
      G = Y- (U+V)/4
      B = U + G 
    """
def FromYUV(data:np.array ):
    b = np.empty_like(a)
    for i in range(data.shape[0]):
       for j in range(data.shape[1]):
         b[i][j][1] = data[i][j][0] - (data[i][j][1] + data[i][j][2])/4

         b[i][j][0] = b[i][j][1] + data[i][j][1]
         b[i][j][2] = b[i][j][1] + data[i][j][2]
    return b


def KL(data:np.array):
   pass