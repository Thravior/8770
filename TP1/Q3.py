import numpy as np
import time
import math
from anytree import Node, RenderTree, PreOrderIter, AsciiStyle

folders = [r"texte",r"image"]
extension = {"image":"png","texte":"txt"}

def Compress(typef, num):
    start_time = time.time()

    b = bytearray()
    file_path =   typef+"s/"+typef+"_"+str(num)+"."+extension[typef]
    with open(file_path, 'rb') as file:
        f = file.read()
        b = bytearray(f)

    #Message à coder
    Message = b
    diff = 1 
    if b[0:9].hex() == '89504e470d0a1a0a00': # est png
        if b[25] == 2:
            diff = 3
        elif b[25] == 6:
            diff = 4
        elif b[25] == 4:
            diff = 2

    l=[]
    for i in range(len(Message)):
        if diff == 0:
            break
        if i<diff:
            l.append(Message[i])
        else:
            l.append((Message[i]^Message[i-diff]))

    if len(l)>0:
        Message = l 
    #Préparation pour la création de l'arbre. On trouve les feuilles (symboles) et leurs poids (nb occurences).

    #Liste qui sera modifié jusqu'à ce qu'elle contienne seulement la racine de l'arbre
    ArbreSymb =[[hex(Message[0]), Message.count(Message[0]), Node(hex(Message[0]))]]
    #dictionnaire obtenu à partir de l'arbre.
    dictionnaire = [[hex(Message[0]), '']]
    nbsymboles = 1

    #Recherche des feuilles de l'arbre
    for i in range(1,len(Message)):
        if not list(filter(lambda x: x[0] == hex(Message[i]), ArbreSymb)):
            ArbreSymb += [[hex(Message[i]), Message.count(Message[i]),Node(hex(Message[i]))]]
            dictionnaire += [[hex(Message[i]), '']]
            nbsymboles += 1

    longueurOriginale = np.ceil(np.log2(nbsymboles))*len(Message)

    OccSymb = ArbreSymb.copy()

    ArbreSymb = sorted(ArbreSymb, key=lambda x: x[1])
    #Création de l'arbre

    while len(ArbreSymb) > 1:
        #Fusion des noeuds de poids plus faibles
        symbfusionnes = ArbreSymb[0][0] + ArbreSymb[1][0]
        #Création d'un nouveau noeud
        noeud = Node(symbfusionnes)
        temp = [symbfusionnes, ArbreSymb[0][1] + ArbreSymb[1][1], noeud]
        #Ajustement de l'arbre pour connecter le nouveau avec ses parents
        ArbreSymb[0][2].parent = noeud
        ArbreSymb[1][2].parent = noeud
        #Enlève les noeuds fusionnés de la liste de noeud à fusionner.
        del ArbreSymb[0:2]
        #Ajout du nouveau noeud à la liste et tri.
        ArbreSymb += [temp]
        #Pour affichage de l'arbre ou des sous-branches
        ArbreSymb = sorted(ArbreSymb, key=lambda x: x[1])

    ArbreCodes = Node('')
    noeud = ArbreCodes
    #print([node.name for node in PreOrderIter(ArbreSymb[0][2])])
    parcoursprefix = [node for node in PreOrderIter(ArbreSymb[0][2])]
    parcoursprefix = parcoursprefix[1:len(parcoursprefix)] #ignore la racine

    Prevdepth = 0 #pour suivre les mouvements en profondeur dans l'arbre
    for node in parcoursprefix:  #Liste des noeuds
        if Prevdepth < node.depth: #On va plus profond dans l'arbre, on met un 0
            temp = Node(noeud.name + '0')
            noeud.children = [temp]
            if node.children: #On avance le "pointeur" noeud si le noeud ajouté a des enfants.
                noeud = temp
        elif Prevdepth == node.depth: #Même profondeur, autre feuille, on met un 1
            temp = Node(noeud.name + '1')
            noeud.children = [noeud.children[0], temp]  #Ajoute le deuxième enfant
            if node.children: #On avance le "pointeur" noeud si le noeud ajouté a des enfants.
                noeud = temp
        else:
            for i in range(Prevdepth-node.depth): #On prend une autre branche, donc on met un 1
                noeud = noeud.parent #On remontre dans l'arbre pour prendre la prochaine branche non explorée.
            temp = Node(noeud.name + '1')
            noeud.children = [noeud.children[0], temp]
            if node.children:
                noeud = temp

        Prevdepth = node.depth

    ArbreSymbList = [node for node in PreOrderIter(ArbreSymb[0][2])]
    ArbreCodeList = [node for node in PreOrderIter(ArbreCodes)]

    for i in range(len(ArbreSymbList)):
        if ArbreSymbList[i].is_leaf: #Génère des codes pour les feuilles seulement
            temp = list(filter(lambda x: x[0] == ArbreSymbList[i].name, dictionnaire))
            if temp:
                indice = dictionnaire.index(temp[0])
                dictionnaire[indice][1] = ArbreCodeList[i].name

    MessageCode = []
    longueur = 0
    for i in range(len(Message)):
        substitution = list(filter(lambda x: x[0] == hex(Message[i]), dictionnaire))
        MessageCode += [substitution[0][1]]
        longueur += len(substitution[0][1])

#    print('Espérance: ' + str(longueur/len(Message)))
    entropie =0
    for i in range(nbsymboles):
        entropie = entropie-(OccSymb[i][1]/len(Message))*math.log(OccSymb[i][1]/len(Message),2)

#    print('Entropie: ' + str(entropie))
    return (1-longueur/longueurOriginale, time.time()-start_time)

for folder in folders:
    for i in range(1,6):
        ratio, duree = Compress(folder,i)
        print((folder+"_"+str(i)+"."+extension[folder]+ ": \n\tRatio: " + "{:.5f}".format(ratio) + "\n\tTemps de compression: " + "{:.5f}".format(duree) ))
