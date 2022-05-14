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


//

//


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