from PIL import Image
import cv2
import numpy as np
from numpy import linalg as LA


"""
  Y = (R + 2G + B) /4
  U = B − G
  V = R − G 
"""
def ToYUV(data:np.array ):
    b = np.copy(data)
    for i in range(data.shape[0]):
       for j in range(data.shape[1]):
         b[i][j][0] = (int(data[i][j][0]) + 2*int(data[i][j][1]) + int(data[i][j][2]))/4          
         b[i][j][1] = int(data[i][j][2]) - int(data[i][j][1])
         b[i][j][2] = int(data[i][j][0]) - int(data[i][j][1])
    return b

"""
  R = V + G
  G = Y - (U+V)/4
  B = U + G 
"""
def FromYUV(data:np.array ):
    b = np.copy(data)
    for i in range(data.shape[0]):
       for j in range(data.shape[1]):
         b[i][j][1] = data[i][j][0] - (data[i][j][1] + data[i][j][2])/4

         b[i][j][2] = b[i][j][1] + data[i][j][1]
         b[i][j][0] = b[i][j][1] + data[i][j][2]
    return b



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
  imageKL = np.copy(image)  

  vecMoy =[[Moyenne[0]], [Moyenne[1]], [Moyenne[2]]] 

  for i in range(len(image)):
      for j in range(len(image[0])):
          vecTemp=[[image[i][j][0]], [image[i][j][1]], [image[i][j][2]]]
          #a=Mb
          imageKL[i][j][:] = np.reshape(np.dot(eigvec,np.subtract(vecTemp,vecMoy)),(3))
  return (imageKL, eigvec, Moyenne)

def KLInverse(image:np.array, eigvec:np.array, Moyenne:np.array):
  # Inverse
  invEigvec = LA.pinv(eigvec)

  vecMoy =[Moyenne[0], Moyenne[1], Moyenne[2]] 
  imageTR = np.copy(image)

  for i in range(len(image)):
      for j in range(len(image[0])):
        #b=inv(M)a
            vecTemp=[[imageKL[i][j][0]], [imageKL[i][j][1]], [imageKL[i][j][2]]]
            imageTR[i][j][:] = np.add(np.reshape(np.dot(invEigvec,vecTemp),(3)),vecMoy)
  return imageTR

"""
suppose entrée sur 8 bits et sorties sur values bits
"""
def Quantify(data:np.array, values:tuple[int,int,int]):
    values = [8-i for i in values]
    dataNew = data.astype('uint8')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j][0] = dataNew[i][j][0] >> int(values[0])
            data[i][j][1] = dataNew[i][j][1] >> int(values[1])
            data[i][j][2] = dataNew[i][j][2] >> int(values[2])

"""
suppose entrées sur values bit et sortie sur 8 bits 
"""
def Unquantify(data:np.array, values:tuple[int,int,int]):
    values = [8-i for i in values]
    dataNew = data.astype('uint8')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j][0] = dataNew[i][j][0] << int(values[0])
            data[i][j][1] = dataNew[i][j][1] << int(values[1])
            data[i][j][2] = dataNew[i][j][2] << int(values[2])

with Image.open(r"TP2\\data\\kodim01.png") as im:
  """  array = np.asarray(im) 
  imageYUV = ToYUV(array)
  imageKL, vecteur, Moyenne = KL(imageYUV)
  original = KLInverse(imageKL,vecteur,Moyenne)
  print(imageYUV[2][4])
  print(imageKL[2][4])
  print(original[2][4])
  """
  
  import matplotlib.pyplot as py  
  fig1 = py.figure(figsize = (10,10))
  imagelue = np.asarray(im)
#  py.imread(r"TP2\\data\\kodim01.png")

  imageC=imagelue.astype('double')
  image = ToYUV(imageC)

  imageKL, eigvec, Moyenne = KL(image)

  Quantifications = [((8,8,8),np.copy(imageKL)),((8,8,4),np.copy(imageKL)),((8,8,0),np.copy(imageKL)),((8,6,2),np.copy(imageKL))]
  for Values in Quantifications:
      Quantify(Values[1],Values[0])
      Unquantify(Values[1],Values[0])
      KLInverse

  results = []
  for Values in Quantifications:
      results.append(FromYUV(KLInverse(Values[1],eigvec,Moyenne)))
  for result in results:      
    fig2 = py.figure(figsize = (10,10))
    imageout = np.clip(result,0,255)
    imageout= imageout.astype('uint8')
    py.imshow(imageout)
    py.show()