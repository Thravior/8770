import numpy as np


b = bytearray()
file_path = "images/image_5.png"
with open(file_path, 'rb') as file:
    f = file.read()
    b = bytearray(f)

m = "ABAABAABACABBABCDAADACABABAAABAABBABABAABAABAABAABAABAABAAB"

encoded=bytearray(m.encode('utf-8'))

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
dictionnaire = np.transpose([dictsymb,dictbin])
print(dictionnaire)
print("Longueur dictionnaire:" +str(len(dictsymb)))


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
    a = dictsymb.index(precsouschaine)
    index = dictbin[a]
    codebinaire = [dictbin[a]]

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
print(MessageCode)
#Affichage du dictionnaire final


#print(a)
dictionnaire = np.transpose([dictsymb,dictbin])
print(dictionnaire)
print("Longueur dictionnaire:" +str(len(dictsymb)))

print("Longueur = {0}".format(longueur))
print("Longueur originale = {0}".format(longueurOriginale))
