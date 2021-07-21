# coding=utf-8

# image verification

import sys
import numpy as np
from scipy.spatial.distance import cosine
from keras.preprocessing import image
#from senet import get_features_array
from resnet import get_features_array
#from inception import get_features_array


def load_img(filename, required_size):
    img = image.load_img(filename, target_size=required_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


# 返回图片的特征值
def get_features(filename, required_size=(224, 224)): 
    x = load_img(filename, required_size=required_size)
    yhat2 = get_features_array(x)
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

    x = []

    x.append( get_features('data/1.jpg') )
    x.append( get_features('data/2.jpg') )
    x.append( get_features('data/3.jpg') )
    x.append( get_features('data/4.png') )
    x.append( get_features('data/5.jpg') )
    x.append( get_features('data/6.jpg') )
    x.append( get_features('data/9.jpg') )

    for i in range(len(x)):
        for j in range(len(x)):
            if i==j:
                f1 = 0.0
            else:
                ed = euclidean_distance(x[i], x[j])[0]
                cd = cosine_distance(x[i], x[j])
                f1 = 2*ed*cd/(ed+cd)
            print('%.6f'%f1, end='\t')
        print('')


'''
inception

euclidean_distance: 
0.000000    0.450301    0.574838    0.721270    0.776129    0.447674    0.820329    
0.450301    0.000000    0.634814    0.730928    0.786120    0.517298    0.882757    
0.574838    0.634814    0.000000    0.768758    0.864813    0.635137    0.897393    
0.721270    0.730928    0.768758    0.000000    0.659823    0.705663    0.913525    
0.776129    0.786120    0.864813    0.659823    0.000000    0.775498    0.924396    
0.447674    0.517298    0.635137    0.705663    0.775498    0.000000    0.828087    
0.820329    0.882757    0.897393    0.913525    0.924396    0.828087    0.000000    
cosine_distance: 
0.000000    0.101386    0.165220    0.260115    0.301188    0.100206    0.336470    
0.101386    0.000000    0.201495    0.267128    0.308993    0.133799    0.389630    
0.165220    0.201495    0.000000    0.295495    0.373950    0.201700    0.402658    
0.260115    0.267128    0.295495    0.000000    0.217683    0.248980    0.417264    
0.301188    0.308993    0.373950    0.217683    0.000000    0.300699    0.427253    
0.100206    0.133799    0.201700    0.248980    0.300699    0.000000    0.342864    
0.336470    0.389630    0.402658    0.417264    0.427253    0.342864    0.000000    

ResNet

euclidean_distance: 
0.000000    0.512030    0.695662    0.961611    1.014048    0.621335    1.140669    
0.512030    0.000000    0.588233    0.960121    1.007514    0.529340    1.160422    
0.695662    0.588233    0.000000    1.006610    1.040069    0.689072    1.196737    
0.961611    0.960121    1.006610    0.000000    1.013785    0.930929    1.218106    
1.014048    1.007514    1.040069    1.013785    0.000000    1.007207    1.205612    
0.621335    0.529340    0.689072    0.930929    1.007207    0.000000    1.171229    
1.140669    1.160422    1.196737    1.218106    1.205612    1.171229    0.000000    
cosine_distance: 
0.000000    0.131088    0.241973    0.462347    0.514147    0.193028    0.650563    
0.131088    0.000000    0.173009    0.460916    0.507542    0.140100    0.673290    
0.241973    0.173009    0.000000    0.506632    0.540871    0.237410    0.716089    
0.462347    0.460916    0.506632    0.000000    0.513881    0.433314    0.741891    
0.514147    0.507542    0.540871    0.513881    0.000000    0.507233    0.726750    
0.193028    0.140100    0.237410    0.433314    0.507233    0.000000    0.685889    
0.650563    0.673290    0.716089    0.741891    0.726750    0.685889    0.000000

ResNet F1

0.000000    0.208736    0.359055    0.624454    0.682334    0.294550    0.828566    
0.208736    0.000000    0.267378    0.622834    0.675032    0.221561    0.852152    
0.359055    0.267378    0.000000    0.674024    0.711657    0.353148    0.896025    
0.624454    0.622834    0.674024    0.000000    0.682040    0.591368    0.922146    
0.682334    0.675032    0.711657    0.682040    0.000000    0.674690    0.906847    
0.294550    0.221561    0.353148    0.591368    0.674690    0.000000    0.865139    
0.828566    0.852152    0.896025    0.922146    0.906847    0.865139    0.000000    
'''
