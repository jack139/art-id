# coding=utf-8

# face verification with the VGGFace2 model

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import sys
import numpy as np
from scipy.spatial.distance import cosine
from keras.preprocessing import image
from keras.applications import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
import face_recognition

import tensorflow as tf

INPUT_SIZE = (224, 224, 3)

graph = tf.Graph()  # 解决多线程不同模型时，keras或tensorflow冲突的问题
session = tf.Session(graph=graph)
with graph.as_default():
    with session.as_default():
        base_model = InceptionResNetV2(weights='imagenet', input_shape=INPUT_SIZE, include_top=False)
        # add a global spatial average pooling layer
        model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))
        # https://stackoverflow.com/questions/40850089/is-keras-thread-safe
        model._make_predict_function() # have to initialize before threading


def load_img(filename, required_size):
    img = image.load_img(filename, target_size=required_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


# 返回图片的特征值
def get_features(filename, required_size=(224, 224)): 
    # extract faces
    x = load_img(filename, required_size=required_size)
    yhat2 = get_features_array(x)
    return yhat2

# 根据人脸列表返回特征
def get_features_array(img):
    # convert into an array of samples
    samples = np.asarray(img, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples)
    # perform prediction
    with graph.as_default(): # 解决多线程不同模型时，keras或tensorflow冲突的问题
        with session.as_default():
            yhat = model.predict(samples)
    yhat2 = yhat / np.linalg.norm(yhat)
    return yhat2

# 特征值距离
def face_distance(encodings, encodings_to_compare):
    if len(encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(encodings - encodings_to_compare, axis=1)

# 特征值余弦距离
def is_match(known_embedding, candidate_embedding):
    return cosine(known_embedding, candidate_embedding)
