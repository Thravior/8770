import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

folders = [r"texte",r"image"]
extension = {"image":"png","texte":"txt"}

def Histo(typef, num):
    b = bytearray()
    file_path =   typef+"s/"+typef+"_"+str(num)+"."+extension[typef]
    with open(file_path, 'rb') as file:
        f = file.read()
        b = bytearray(f)
    Message = b
    
    x=len(Message)
    y = 1
    diff = 1 

    if b[0:9].hex() == '89504e470d0a1a0a00': # est png
        x = int(b[16:20].hex(),16)
        print(int(b[20:24].hex(),16))

        if b[25] == 2:
            diff = 3
        elif b[25] == 6:
            diff = 4
        elif b[25] == 4:
            diff = 2



    l=[]
    for i in range(len(Message)):
        if i<diff:
            l.append(Message[i])
        else:
            l.append((Message[i]^Message[i-diff]))

    return (l,257)


for folder in folders:
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 6))
    fig.suptitle('Histogrammes', fontsize=16)
    for i in range(1,6):        
        ax = axes[ i-1 % 5]
        data, types = Histo(folder, i)

        ax.hist(data, bins=np.arange(min(257, types+1)), edgecolor='black', alpha=0.7)
        ax.set_title(f'{types}:{folder}_{i}.{extension[folder]}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show() 


