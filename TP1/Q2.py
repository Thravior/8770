import numpy as np

folders = [r"texte",r"image"]
extension = {"image":"png","texte":"txt"}
resultats = []

def Compress(typef, num):
    b = bytearray()
    file_path =   "TP1\\"+ typef+"s/"+typef+"_"+str(num)+"."+extension[typef]
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

    #dictsymb.sort()
#    dictionnaire = np.transpose([dictsymb,dictbin])

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

    ## Tests
#    print(MessageCode)

    #Affichage du dictionnaire final


    #print(a)
    #dictionnaire = np.transpose([dictsymb,dictbin])

    #print("Longueur = {0}".format(longueur))
    #print("Longueur originale = {0}".format(longueurOriginale))
    return 1-longueur/longueurOriginale 


for folder in folders:
    for i in range(1,6):
        resultats.append((folder+"_"+str(i)+"."+extension[folder], Compress(folder,i)))
print(resultats)