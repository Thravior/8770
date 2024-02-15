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
    b = np.empty_like(data)
    for i in range(data.shape[0]):
       for j in range(data.shape[1]):
         b[i][j][1] = data[i][j][0] - (data[i][j][1] + data[i][j][2])/4

         b[i][j][0] = b[i][j][1] + data[i][j][1]
         b[i][j][2] = b[i][j][1] + data[i][j][2]
    return b



def KL(data:np.array):
    Moy = np.array([0.0,0.0,0.0])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
           Moy += data[i][j]
    Moy/=(data.shape[0]*data.shape[1])
    print(Moy)
    """Github du Cours"""
    covRGB = np.zeros((3,3), dtype = "double")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
          vecTemp=[[data[i][j][0] - Moy[0]], [data[i][j][1] - Moy[1]], [data[i][j][2] - Moy[2]] ]
          vecProdTemp = np.dot(vecTemp,np.transpose(vecTemp))
          covRGB = np.add(covRGB,vecProdTemp)
    covRGB /=(data.shape[0]*data.shape[1])
    print(covRGB)
    eigval, eigvec = LA.eig(covRGB)
    print(eigval)
    print(eigvec)

    # 
    eigvec = np.transpose(eigvec)

    imageKL = np.copy(data)

    vecMoy =[[Moy[0]], [Moy[1]], [Moy[2]]] 

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            vecTemp=[[data[i][j][0]], [data[i][j][1]], [data[i][j][2]]]
            #a=Mb
            imageKL[i][j][:] = np.reshape(np.dot(eigvec,np.subtract(vecTemp,vecMoy)),(3))
        
    return (imageKL, eigvec, Moy)


"""
suppose entrée et sortie sur 8 bits, entree devient sur values bits
"""
def Quantify(data:np.array, values:tuple[int,int,int]):
    values = [8-i for i in values]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j][0] = data[i][j][0] >> values[0]  
            data[i][j][0] = data[i][j][0] >> values[1]
            data[i][j][0] = data[i][j][0] >> values[2]

"""
suppose entrée et sortie sur 8 bits, entree devient 
"""
def Unquantify(data:np.array, values:tuple[int,int,int]):
    values = [8-i for i in values]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j][0] = data[i][j][0] << values[0]  
            data[i][j][0] = data[i][j][0] << values[1]
            data[i][j][0] = data[i][j][0] << values[2]

with Image.open(r"TP2\\data\\kodim0"+str(i) +".png") as im:
  red = list(im.getdata(0))
  green = list(im.getdata(1))
  blue = list(im.getdata(2))
  array = np.asarray(im)    
  imageKL, vecteur, Moyeenne = KL(array)
