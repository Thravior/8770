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
    b = np.empty_like(data,dtype= int)
    for i in range(data.shape[0]):
       for j in range(data.shape[1]):
         b[i][j][0] = (int(data[i][j][0]) + 2*int(data[i][j][1]) + int(data[i][j][2]))/4          
         b[i][j][1] = int(data[i][j][2]) - int(data[i][j][1])
         b[i][j][2] = int(data[i][j][0]) - int(data[i][j][1])
    return b

"""
  R = V + G
  G = Y- (U+V)/4
  B = U + G 
"""
def FromYUV(data:np.array ):
    b = np.empty_like(data)
    for i in range(data.shape[0]):
       for j in range(data.shape[1]):
         b[i][j][1] = data[i][j][0] - (data[i][j][1] + data[i][j][2])/4

         b[i][j][0] = b[i][j][1] + data[i][j][1]
         b[i][j][2] = b[i][j][1] + data[i][j][2]
    return b



def KL(data:np.array):
    # Moyenne
    pixels = 0
    Moy = np.array([0.0,0.0,0.0])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
           Moy += data[i][j]
           pixels += 1
    Moy /= pixels
    print(Moy)

    # Covariance
    """Github du Cours"""
    covRGB = np.zeros((3,3), dtype = "double")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
          vecTemp=[[data[i][j][0] - Moy[0]], [data[i][j][1] - Moy[1]], [data[i][j][2] - Moy[2]] ]
          vecProdTemp = np.dot(vecTemp,np.transpose(vecTemp))
          covRGB = np.add(covRGB,vecProdTemp)
    covRGB /= pixels
    print(covRGB)

    # Vecteur et valeur propres
    eigval, eigvec = LA.eig(covRGB)
    print(eigval)
    print(eigvec)
    # 
    eigvec = np.transpose(eigvec)
    print(eigvec)

    imageKL = np.empty_like(data)

    vecMoy =[[Moy[0]], [Moy[1]], [Moy[2]]] 

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            vecTemp=[[data[i][j][0]], [data[i][j][1]], [data[i][j][2]]]
            #a=Mb
            imageKL[i][j][:] = np.reshape(np.dot(eigvec,np.subtract(vecTemp,vecMoy)),(3))
        
    return (imageKL, eigvec, Moy)

def KLInverse(data:np.array, vector:np.array, Moy:np.array):
    inv = LA.pinv(vector)
    vecMoy =[[Moy[0]], [Moy[1]], [Moy[2]]] 
    imageOri = np.empty_like(data)

    for i in range(data.shape[0]):
      for j in range(data.shape[1]):
        vecTemp=[[data[i][j][0] - Moy[0]], [data[i][j][1] - Moy[1]], [data[i][j][2] - Moy[2]] ]
        imageOri[i][j][:] = np.reshape(np.dot(inv,np.subtract(vecTemp,vecMoy)),(3))
    return imageOri

"""
suppose entrée sur 8 bits et sorties sur values bits
"""
def Quantify(data:np.array, values:tuple[int,int,int]):
    values = [8-i for i in values]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j][0] = data[i][j][0] >> values[0]  
            data[i][j][1] = data[i][j][1] >> values[1]
            data[i][j][2] = data[i][j][2] >> values[2]

"""
suppose entrées sur values bit et sortie sur 8 bits 
"""
def Unquantify(data:np.array, values:tuple[int,int,int]):
    values = [8-i for i in values]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j][0] = data[i][j][0] << values[0]  
            data[i][j][1] = data[i][j][1] << values[1]
            data[i][j][2] = data[i][j][2] << values[2]

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
  imagelue = py.imread(r"TP2\\data\\kodim01.png")
  image=imagelue.astype('double')

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

  invEigvec = LA.pinv(eigvec)

  vecMoy =[Moyenne[0], Moyenne[1], Moyenne[2]] 
  imageRGB = np.copy(image)

  for i in range(len(image)):
      for j in range(len(image[0])):
        #b=inv(M)a
            vecTemp=[[imageKL[i][j][0]], [imageKL[i][j][1]], [imageKL[i][j][2]]]
            imageRGB[i][j][:] = np.add(np.reshape(np.dot(invEigvec,vecTemp),(3)),vecMoy)
            print(str(image[i][j]) + " "+ str(imageRGB[i][j]) )