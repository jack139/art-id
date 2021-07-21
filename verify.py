# coding=utf-8

# image verification

import sys
import numpy as np
from scipy.spatial.distance import cosine
from keras.preprocessing import image
from senet import get_features_array as get_features_array_senet
from resnet import get_features_array as get_features_array_resnet
from inception import get_features_array as get_features_array_inception


def load_img(filename, required_size):
    img = image.load_img(filename, target_size=required_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


# 返回图片的特征值
def get_features(filename, required_size=(224, 224), model_type="senet"): 
    # extract faces
    x = load_img(filename, required_size=required_size)
    if model_type=='senet':
        yhat2 = get_features_array_senet(x)
    elif model_type=='resnet':
        yhat2 = get_features_array_resnet(x)
    else:
        yhat2 = get_features_array_inception(x)
    return yhat2


# 特征值距离
def euclidean_distance(encodings, encodings_to_compare):
    if len(encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(encodings - encodings_to_compare, axis=1)


# 特征值余弦距离
def cosine_distance(known_embedding, candidate_embedding):
    return cosine(known_embedding, candidate_embedding)


if __name__ == '__main__':
    print('\nSENet model')

    a = get_features('data/1.jpg', model_type='senet')
    b = get_features('data/2.jpg', model_type='senet')
    c = get_features('data/6.jpg', model_type='senet')

    print("DIS. a - b : ", euclidean_distance(a, b))
    print("DIS. a - c : ", euclidean_distance(a, c))
    print("DIS. b - c : ", euclidean_distance(b, c))

    print("COS. a - b : ", cosine_distance(a, b))
    print("COS. a - c : ", cosine_distance(a, c))
    print("COS. b - c : ", cosine_distance(b, c))

    print('\nResNet model')

    a = get_features('data/1.jpg', model_type='resnet')
    b = get_features('data/2.jpg', model_type='resnet')
    c = get_features('data/6.jpg', model_type='resnet')

    print("DIS. a - b : ", euclidean_distance(a, b))
    print("DIS. a - c : ", euclidean_distance(a, c))
    print("DIS. b - c : ", euclidean_distance(b, c))

    print("COS. a - b : ", cosine_distance(a, b))
    print("COS. a - c : ", cosine_distance(a, c))
    print("COS. b - c : ", cosine_distance(b, c))

    print('\nInceptionResNetV2 model')

    a = get_features('data/1.jpg', model_type='inception')
    b = get_features('data/2.jpg', model_type='inception')
    c = get_features('data/6.jpg', model_type='inception')

    print("DIS. a - b : ", euclidean_distance(a, b))
    print("DIS. a - c : ", euclidean_distance(a, c))
    print("DIS. b - c : ", euclidean_distance(b, c))

    print("COS. a - b : ", cosine_distance(a, b))
    print("COS. a - c : ", cosine_distance(a, c))
    print("COS. b - c : ", cosine_distance(b, c))
