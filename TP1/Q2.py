import numpy as np
import time

folders = [r"texte",r"image"]
extension = {"image":"png","texte":"txt"}

def Compress(typef, num):
    start_time = time.time()

    b = bytearray()
    file_path =   typef+"s/"+typef+"_"+str(num)+"."+extension[typef]
    with open(file_path, 'rb') as file:
        f = file.read()
        b = bytearray(f)

    Message = b

    # initialisation:
    dictsymb =[hex(Message[0])]
    dictbin = ["{:b}".format(0)]
    nbsymboles = 1
    for i in range(1,len(Message)):
        if hex(Message[i]) not in dictsymb:
            dictsymb += [hex(Message[i])]
            dictbin += ["{:b}".format(nbsymboles)] 
            nbsymboles +=1
            if nbsymboles > 255: #considère des octets et aurait donc tous les octets
                break

    longueurOriginale = np.ceil(np.log2(nbsymboles))*len(Message)    

    # ajustements initiaux
    for i in range(nbsymboles):
        dictbin[i] = "{:b}".format(i).zfill(int(np.ceil(np.log2(nbsymboles))))

    # Codage:
    i=0
    MessageCode = []
    longueur = 0
    while i < len(Message):
        precsouschaine = hex(Message[i]) #sous-chaine qui sera codé
        souschaine = hex(Message[i]) #sous-chaine qui sera codé + 1 caractère (pour le dictionnaire)
        
        #Cherche la plus grande sous-chaine. On ajoute un caractère au fur et à mesure.
        while souschaine in dictsymb and i < len(Message):
            i += 1
            precsouschaine = souschaine
            if i < len(Message):  #Si on a pas atteint la fin du message
                souschaine += hex(Message[i])

        #Codage de la plus grande sous-chaine à l'aide du dictionnaire  
        index = dictsymb.index(precsouschaine)
        codebinaire = [dictbin[index]]

        MessageCode += codebinaire
        longueur += len(codebinaire[0]) 

        #Ajout de la sous-chaine codé + symbole suivant dans le dictionnaire.
        if i < len(Message):
            dictsymb += [souschaine]
            dictbin += ["{:b}".format(nbsymboles)] 
            nbsymboles +=1
        
        #Ajout de 1 bit si requis
        if np.ceil(np.log2(nbsymboles)) > len(MessageCode[-1]):
            for j in range(nbsymboles):
                dictbin[j] = "{:b}".format(j).zfill(int(np.ceil(np.log2(nbsymboles))))


    return (1-longueur/longueurOriginale, time.time()-start_time)


for folder in folders:
    for i in range(1,6):
        ratio, duree = Compress(folder,i)
        print((folder+"_"+str(i)+"."+extension[folder]+ ": \n\tRatio: " + "{:.5f}".format(ratio) + "\n\tTemps de compression: " + "{:.5f}".format(duree) ))
