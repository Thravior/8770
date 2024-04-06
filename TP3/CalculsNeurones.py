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
import sklearn.metrics.pairwise import cosine_similarity 

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

def affinity_mesure(vector1, vector2):
    return cosine_similarity(vector1, vector2)

def search_video(query_image, database, model, layer, affinity_measure, num_keyframes = 10, threshold = 0.8):
    query_vector = encode_image(query_image, model, layer)
    best_match = None
    best_similarity = -1

    for video_name, keyframes in database.items():
        selected_keyframes = keyframes[:num_keyframes]
        for frame_index, frame_vector in enumerate(selected_keyframes):
            similarity = affinity_mesure(query_vector, frame_vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (video_name, frame_index)

    if best_similarity >= threshold:
        return best_match
    else:
        return("out", None)