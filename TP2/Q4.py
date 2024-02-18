from PIL import Image
import cv2
import numpy as np
from numpy import linalg as LA
import math
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import copy
0
"""
  Y = (R + 2G + B) /4
  U = B − G
  V = R − G 
"""
def ToYUV(data:np.array ):
    b = np.copy(data).astype('double')
    for i in range(data.shape[0]):
       for j in range(data.shape[1]):
         b[i][j][0] = (int(data[i][j][0]) + 2*int(data[i][j][1]) + int(data[i][j][2]))/4          
         b[i][j][1] = int(data[i][j][2]) - int(data[i][j][1])
         b[i][j][2] = int(data[i][j][0]) - int(data[i][j][1])
    return b.astype("double")

"""
  R = V + G
  G = Y - (U+V)/4
  B = U + G 
"""
def FromYUV(data:np.array ):
    b = copy.deepcopy(data).astype('double')
    for i in range(data.shape[0]):
       for j in range(data.shape[1]):
         b[i][j][1] = data[i][j][0] - (data[i][j][1] + data[i][j][2])/4

         b[i][j][2] = b[i][j][1] + data[i][j][1]
         b[i][j][0] = b[i][j][1] + data[i][j][2]
    return b.astype('uint8')



def KL(image:np.array):
  Moyenne = np.array([0.0,0.0,0.0])

  for i in range(len(image)):
      for j in range(len(image[0])):
          Moyenne[0]+= image[i][j][0]
          Moyenne[0]+= image[i][j][1]
          Moyenne[0]+= image[i][j][2]
          
  nbPixels = len(image)*len(image[0])        

  Moyenne[0] /= nbPixels
  Moyenne[1] /= nbPixels
  Moyenne[2] /= nbPixels

  covRGB = np.zeros((3,3), dtype = "double")
  for i in range(len(image)):
      for j in range(len(image[0])):
          vecTemp=[[image[i][j][0] - Moyenne[0]], [image[i][j][1]] - Moyenne[1], [image[i][j][2] - Moyenne[2]]]
          vecProdTemp = np.dot(vecTemp,np.transpose(vecTemp))
          covRGB = np.add(covRGB,vecProdTemp)

  covRGB = covRGB / nbPixels        

  eigval, eigvec = LA.eig(covRGB)

  eigvec = np.transpose(eigvec)
  imageKL = np.copy(image).astype('double')

  vecMoy =[[Moyenne[0]], [Moyenne[1]], [Moyenne[2]]] 

  for i in range(len(image)):
      for j in range(len(image[0])):
          vecTemp=[[image[i][j][0]], [image[i][j][1]], [image[i][j][2]]]
          #a=Mb
          imageKL[i][j][:] = np.reshape(np.dot(eigvec,np.subtract(vecTemp,vecMoy)),(3))
  return (imageKL, eigvec, Moyenne)

def KLpartiel(imageIN:np.array, eigvecIN:np.array, MoyenneIN:np.array):
  imageKLpart = np.copy(imageIN).astype('double')
  vecMoy =[[MoyenneIN[0]], [MoyenneIN[1]], [MoyenneIN[2]]] 
  for i in range(len(imageKLpart)):
      for j in range(len(imageKLpart[0])):
          vecTemp=[[imageIN[i][j][0]], [imageIN[i][j][1]], [imageIN[i][j][2]]]
          #a=Mb
          imageKLpart[i][j][:] = np.reshape(np.dot(eigvecIN,np.subtract(vecTemp,vecMoy)),(3))
  return imageKLpart
   

def KLInverse(image:np.array, eigvec:np.array, Moyenne:np.array):
  # Inverse
  invEigvec = LA.pinv(eigvec)

  vecMoy =[Moyenne[0], Moyenne[1], Moyenne[2]] 
  imageTR = copy.deepcopy(np.copy(image))

  for i in range(len(image)):
      for j in range(len(image[0])):
        #b=inv(M)a
            vecTemp=[[image[i][j][0]], [image[i][j][1]], [image[i][j][2]]]
            imageTR[i][j][:] = np.add(np.reshape(np.dot(invEigvec,vecTemp),(3)),vecMoy)
  return imageTR.astype("int8")

"""
suppose entrée sur 8 bits et sorties sur values bits
"""
def Quantify(data:np.array, values:tuple[int,int,int]):
    dataNew = np.copy(data).astype("double")
    minimun = data.min()
    maximum = data.max()
    intervale = maximum - minimun
    intervaleQuantifie = intervale/64

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for channel in range(data.shape[2]):
                dataNew[i][j][channel] = int((dataNew[i][j][channel]-minimun) / intervaleQuantifie) * intervaleQuantifie + minimun
    return dataNew 

import matplotlib.pyplot as py

liste_images = [r"kodim01.png",r"kodim02.png",r"kodim05.png",r"kodim13.png",r"kodim23.png"]
for Pathimage in liste_images:
    image1 = []
    vecteur1 = None
    Moyenne1 = None
    with Image.open(r"TP2\\data\\"+Pathimage) as im:
        image1 = np.asarray(im) 
        
        _, vecteur1,  Moyenne1 = KL(ToYUV(image1))

        for PathimageSec in liste_images:
            print(Pathimage + " + " + PathimageSec)
            image2 = None
            Originale = None
            with Image.open(r"TP2\\data\\"+PathimageSec) as im2:
                image2 = np.asarray(im2)
                Originale = copy.deepcopy(image2)
                image2 = ToYUV(image2)
                image2KL = KLpartiel(image2,vecteur1,Moyenne1)
                imageComposite = Quantify(image2KL, (5,5,5))

                imageCompositeYUV = KLInverse(imageComposite,vecteur1,Moyenne1)

                imageCompositeRGB = FromYUV(imageCompositeYUV)

                print("\tPSNR:\t", psnr(Originale,imageCompositeRGB))
                print("\tSSIM:\t",ssim(Originale,imageCompositeRGB,channel_axis=2))