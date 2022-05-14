from collections import deque
from glob import glob
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle 

GLOBAL_CONFIG = {'SAMPLE_SZ':(64,64) ,
          'COLORSPACE':'YCrCb',
          'SPATIAL_BIN_SZ':(16,16),
          'COLOR_BINS':32,
          'COLOR_VAL_RANGE':(0,256),
          'HOG_CHANNEL':'ALL',
          'HOG_ORIENTS':9,
          'HOG_PIX_PER_CELL':16,
          'HOG_CELLS_PER_BLOCK':1,
          'CELLS_PER_STEP':2,
          'FRAME_HIST_COUNT':15,
          'HEAT_THRESH':6,
          'ROIS':[[(0,1280),(400,700)],[(640,1280),(400,650)],[(900,1280),(400,650)]],
          'SCALES': [1.5,2,2.5],
          'PREDICTION_THRESH':0.7
        }

#load image
def imgread(path):
    return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)

def load_image_data(paths):
    data = []
    for path in paths:
        img_data = imgread(path)
        data.append(img_data)
    return np.array(data,dtype=np.uint8)


def extract_spatial_bin_features(img,size):
          
    return cv2.resize(img,size).flatten()

# extract color histogram features
def extract_color_hist_features(img,nbins,range_vals):
    chan0_hist = np.histogram(img[:,:,0],bins=nbins,range=range_vals)
    chan1_hist = np.histogram(img[:,:,1],bins=nbins,range=range_vals)
    chan2_hist = np.histogram(img[:,:,2],bins=nbins,range=range_vals)
    
    color_hist_features = np.concatenate((chan0_hist[0],
                                         chan1_hist[0],
                                         chan2_hist[0]))
    return color_hist_features

def extract_hog_features(img_channel,nb_orient, 
                         nb_pix_per_cell,
                         nb_cell_per_block, 
                         visualize= False, 
                         ret_vector=True):
    
    if visualize == True:
        features, hog_image = hog(img_channel,orientations=nb_orient,
                                  pixels_per_cell= (nb_pix_per_cell,nb_pix_per_cell),
                                  cells_per_block = (nb_cell_per_block,nb_cell_per_block),
                                  visualize=True,
                                  feature_vector=ret_vector)
        return features, hog_image
    
    else:
        features  = hog(img_channel,orientations=nb_orient,
                                  pixels_per_cell = (nb_pix_per_cell,nb_pix_per_cell),
                                  cells_per_block = (nb_cell_per_block,nb_cell_per_block),
                                  visualize=False,
                                  feature_vector=ret_vector)
        return features
    
    
def cvtColor(img,colorspace:str):
    if colorspace == 'HSV':
        img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        
    elif colorspace == 'HLS':
        img = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
            
    elif colorspace == 'LUV':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        
    elif colorspace == 'YUV':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    
    elif colorspace == 'YCrCb':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            
    else:
        raise Exception("% colorspace is not a valid colorspace"%(colorspace))

    return img


def get_features(img,hog_channel,colorspace):
    
    if colorspace != 'RGB':
        img = cvtColor(img,colorspace)
        
            
    spatial_bin_features = \
    extract_spatial_bin_features(img,size =  GLOBAL_CONFIG['SPATIAL_BIN_SZ'])
    
    color_hist_features  = \
    extract_color_hist_features(img,
                                nbins=GLOBAL_CONFIG['COLOR_BINS'],
                                range_vals=GLOBAL_CONFIG['COLOR_VAL_RANGE'])
    
    if hog_channel == 'ALL':
        hog_features = [ ]
        
        for channel in range(3):
            hog_features.append(
                    extract_hog_features(
                            img[:,:,channel],
                            nb_orient=GLOBAL_CONFIG['HOG_ORIENTS'],
                            nb_pix_per_cell=GLOBAL_CONFIG['HOG_PIX_PER_CELL'],
                            nb_cell_per_block = GLOBAL_CONFIG['HOG_CELLS_PER_BLOCK']))
            
        hog_features = np.ravel(hog_features)
    
    else:
        hog_features = extract_hog_features(img[:,:,hog_channel],
                            nb_orient=GLOBAL_CONFIG['HOG_ORIENTS'],
                            nb_pix_per_cell=GLOBAL_CONFIG['HOG_PIX_PER_CELL'],
                            nb_cell_per_block = GLOBAL_CONFIG['HOG_CELLS_PER_BLOCK'])
    
    return np.concatenate((spatial_bin_features,
                          color_hist_features,
                          hog_features))

## Build dataset

def build_datasets(car_paths,notcar_paths):
    paths = car_paths+notcar_paths
    
    X = []
    for path in tqdm(paths):
        img = imgread(path)
        X.append(get_features(img,
                              hog_channel= GLOBAL_CONFIG['HOG_CHANNEL'],
                              colorspace=GLOBAL_CONFIG['COLORSPACE']))
        
    X = np.reshape(X,[len(paths),-1])
    
    
    y = np.concatenate((np.ones(len(car_paths)),
                       np.zeros(len(notcar_paths))))
    
    
    Scaler_X = StandardScaler().fit(X)    
    
    X_scaled = Scaler_X.transform(X)
    
    X_scaled, y = shuffle(X_scaled,y)
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X_scaled,y,train_size=0.7)
        
    del([X_scaled,y])
    
    with open('train.p','wb') as f:
        train_set = {'data':X_train, 'labels':y_train}
        pickle.dump(train_set,f)
        
    with open('test.p','wb') as f:
        test_set = {'data':X_test, 'labels':y_test}
        pickle.dump(test_set,f)
    
    with open('scaler.p','wb') as f:
        pickle.dump(Scaler_X,f)
    
# get dataset
def get_datasets(force=False):
    
    if (force == True)\
    or not os.path.isfile('train.p')\
    or not os.path.isfile('test.p'):
            
        # Load all image data.
        vehicle_img_path = []
        vehicle_img_path.extend(glob('vehicles/GTI_Far/*.png'))
        vehicle_img_path.extend(glob('vehicles/GTI_Left/*.png'))
        vehicle_img_path.extend(glob('vehicles/GTI_MiddleClose/*.png'))
        vehicle_img_path.extend(glob('vehicles/GTI_Right/*.png'))
        vehicle_img_path.extend(glob('vehicles/KITTI_extracted/*.png'))
          
        non_vehicle_img_path = []
        non_vehicle_img_path.extend(glob('non-vehicles/Extras/*.png'))
        non_vehicle_img_path.extend(glob('non-vehicles/GTI/*.png'))
        
        build_datasets(vehicle_img_path, non_vehicle_img_path)
        
    with open('train.p','rb') as f:
        train_data = pickle.load(f)
        
    with open('test.p','rb') as f:
        test_data = pickle.load(f)
        
    return (train_data,test_data)