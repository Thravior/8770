from PIL import Image
import cv2
import numpy as np
from numpy import linalg as LA

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# https://gist.github.com/nimpy/5b0085075a54ba2e94f2cfabf5a98a57
def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

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
    b = np.copy(data).astype('double')
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
    values_usable = [8-i for i in values]
    dataNew = np.copy(data).astype("int8")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for channel in range(data.shape[2]):                
              dataNew[i][j][channel] = ( dataNew[i][j][channel] >> values_usable[channel] ) << values_usable[channel]
    print("Quantification:")
    return dataNew 

liste_images = [r"TP2\\data\\kodim01.png",r"TP2\\data\\kodim02.png",r"TP2\\data\\kodim05.png",r"TP2\\data\\kodim13.png",r"TP2\\data\\kodim23.png"]
for Pathimage in liste_images:
  print(Pathimage)
  with Image.open(Pathimage) as im:
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

    imageC=imagelue.astype('double')
    image = ToYUV(imageC)

    imageKL, eigvec, Moyenne = KL(image)

    Quantifications = [(8,8,8),(8,8,4) ,(8,8,0),(8,6,2)]
    for Values in Quantifications:
        print("Compression:\t", Values)
        quantified = Quantify(imageKL,Values)

        inverse = KLInverse(quantified,eigvec,Moyenne)
        rgb = FromYUV(inverse)

    break
