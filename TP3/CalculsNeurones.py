import numpy as np
import math
from numpy.linalg import norm
import cv2
from multiprocessing import Process, Queue
import csv
import tensorflow as tf 
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity 

resnet_model = ResNet50(weights= 'imagenet', include_top= False )

#couche de rep à utiliser 
layer_name = 'avg_pool' 
feature_extractor_model = Model(inputs = resnet_model.input, outputs=resnet_model.get_layer(layer_name).output)

def encode_image(image_path, model):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir en RGB (ResNet-50 utilise ce format)
    image = cv2.resize(image, (224, 224))  # Redimensionner l'image à la taille attendue par ResNet-50
    image = preprocess_input(image)  # Prétraiter l'image selon les spécifications de ResNet-50
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension pour correspondre à la forme d'entrée de ResNet-50
    features = model.predict(image)
    return features

def cosine_affinity_mesure(vector1, vector2):
    return cosine_similarity(vector1, vector2)

def search_video(query_image, database, model, layer, affinity_measure, num_keyframes = 10, threshold = 0.8):
    query_vector = encode_image(query_image, model, layer)
    best_match = None
    best_similarity = -1

    for video_name, keyframes in database.items():
        selected_keyframes = keyframes[:num_keyframes]
        for frame_index, frame_vector in enumerate(selected_keyframes):
            similarity = cosine_affinity_mesure(query_vector, frame_vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (video_name, frame_index)

    if best_similarity >= threshold:
        return best_match
    else:
        return("out", None)
    
if __name__ == 'main':

    import time
    now = time.time()

    model = feature_extractor_model  
    affinity_measure = cosine_similarity_measure()  

    # Espace de couleur avec leur seuil
    fonctions_color = [lambda x: cv2.cvtColor(x,cv2.COLOR_BGR2YUV), lambda x: x]
    seuils = [0.6,0.42]
    desc_color = ["YUV", "RGB"]
    # Exctraction des videos
    fonctions_temps = [extract_frames3ps,extract_frames1s]
    desc_temps = ["3ps","1ps"]
    for c in range(len(fonctions_color)):
        f_color = fonctions_color[c]

        for f in range(len(fonctions_temps)):
            f_extract = fonctions_temps[f]

        # 
        if c == 0 and f == 0:
            continue

        captures = []
        
        now = time.time()

        for target in range(1,101):
            video_path = "TP3\moodle\data\mp4\\v" + str(target).zfill(3) +  ".mp4"
            captures.append(f_extract(video_path,f_color))

        temps_indexation = time.time() - now

        print("Durée indexation:")
        print(temps_indexation)

        csv_file_path = "results_"+desc_temps[f]+"s_Coul"+desc_color[c]+"_hist3d_2pieces_taille64_distManh.csv"

        now = time.time()
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(("image","video_pred","minutage_pred"))
            model = feature_extractor_model  
            affinity_measure = cosine_affinity_mesure

            for i in range(0,1000):
                image_path = "TP3\moodle\data\jpeg\i" + str(i).zfill(3) +".jpeg"
                result = search_video(image_path, video_path)

            csv_writer.writerow(("i"+str(i).zfill(3), final[0],final[1]))
            csv_writer.writerow(("indexation", temps_indexation,"Tests",time.time() - now))