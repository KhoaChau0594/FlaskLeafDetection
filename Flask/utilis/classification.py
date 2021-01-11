import matplotlib.pyplot as plt
import numpy as np

from math import ceil
import cv2

from skimage import io, color
from skimage.filters import threshold_otsu
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2gray
from sklearn import cluster
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
import numpy as np

import pandas as pd

import os
import glob
import time
from decimal import *
import operator
import pickle
import math

class Classifier:
    
    def __init__(self, create_mask_model, classify_model):
        self.create_mask_model = create_mask_model
        self.classify_model = classify_model

    def do_classify(self, imgPath):
        imgName = imgPath.split('\\')[-1]

        start = time.time()
        # create ground truth image
        mask = create_mask(self.create_mask_model, imgPath)
        # remove background
        img = remove_background(imgPath, mask)

        ### segment
        img = segment(img)
        segDest = os.path.join(r'.\segmentedImage', imgName)
        cv2.imwrite(segDest, img)

        ### feature extraction
        data = feature_extraction_whole_img_T(img, 'test')

        ### predict
        res = self.classify_model.predict(data.drop(['label', 'light'], axis=1))
        end = time.time()

        ['Apple___Apple_scab' 'Apple___Black_rot' 'Apple___Cedar_apple_rust' 'Apple___healthy']
        if res == 'Apple___Apple_scab':
            res = 'bệnh nấm tảo - scab'
        elif res == 'Apple___Black_rot':
            res = 'bệnh thối đen - black rot'
        elif res == 'Apple___Cedar_apple_rust':
            res = 'bệnh gỉ lá - rust'
        else:
            res = 'lá khỏe mạnh - healthy'
            img = None

        return [res, end-start]
        # {
        #     'result': res, 
        #     'segmented_Image': img, 
        #     'time': end - start
        # }

def adjust(h):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    cl1 = clahe.apply(h)
    return cl1

def constructImage(h,s,v):
    hsv_new = np.dstack((h, s))
    hsv_new = np.dstack((hsv_new, v))
    hsv_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
    return hsv_new

def adjustHSV(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h = adjust(img[:,:,0])
    s = img[:,:,1]
    v = img[:,:,2]
    
    return constructImage(h,s,v)

from skimage.filters import roberts, sobel, scharr, prewitt

erosion_size = 1
def round_to_1(x):
    return round(x, -int(math.floor(math.log10(abs(x)))))

# ### Contrast Enhancement
def adjustConstrast(P):
    Q = np.zeros(P.shape)
    maxValue = np.amax(P)
    minValue = np.amin(P)
    r, h = P.shape
    for i in range(0, r):
        for y in range(0, h):
            Q[i,y] = (P[i,y] - minValue) / (maxValue - minValue)
    return Q

def createBin(num_of_bin):
    bin = {}
    step = Decimal(format(1 / num_of_bin, '.2f'))
    higher_boundary = step
    # create bin
    for i in range(0, num_of_bin):
        bin[higher_boundary.to_eng_string()] = higher_boundary
        higher_boundary += step
    return bin

def divideIntoBucket(data, num_of_bin):
    bin = createBin(100)
    r, c = data.shape
    for i in range(0, r):
        for y in range(0, c):
            v = Decimal(format(0.01 if data[i, y] == 0.00 else data[i,y], '.2f'))
            if v == Decimal(format(0, '.2f')):
                continue
            bin[v.to_eng_string()] += 1
    return bin
        
def create_mask_feature_extraction(img, labeled_img=None):
    df = pd.DataFrame()
    
    i  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    img2 = i[:,:,0]
    img2 = img2.reshape((-1))
    
    df['Original Image'] = img2
    
    num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
    kernels = []
    for theta in range(2):   #Define number of thetas
        theta = theta / 4. * np.pi
        for sigma in (1, 3):  #Sigma with 1 and 3
            for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
                for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
                    gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    #Now filter the image and add values to a new column 
                    fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img
                    num += 1  #Increment for gabor column label
                    
    #CANNY EDGE
    edges = cv2.Canny(img, 100,200)   #Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe

    #ROBERTS EDGE
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1

    #SOBEL
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1

    #SCHARR
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1

    #PREWITT
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1

    #GAUSSIAN with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1

    #GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3

    #MEDIAN with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1

    #VARIANCE with size=3
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img1 = variance_img.reshape(-1)
    df['Variance s3'] = variance_img1  #Add column to original dataframe
    
    if labeled_img is not None:
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
        labeled_img1 = labeled_img.reshape(-1)
        df['Labels'] = labeled_img1
    
    return df

def create_mask(model, file_path):
    test_img = cv2.imread(file_path)
    r,h,_ = test_img.shape
    data = create_mask_feature_extraction(test_img)
    result = model.predict(data)
    segmented = result.reshape((r,h))
    segmented = cv2.medianBlur(segmented, 7)
    return segmented

def remove_background(img_path, mask):
    img = cv2.imread(img_path)
    r,h,_ = img.shape
    
    for i in range(0,r):
        for y in range(0,h):
            if mask[i, y] <= 2:
                img[i, y, 0] = 0
                img[i, y, 1] = 0
                img[i, y, 2] = 0
    return img
    
def segment(img):
    erosion_size = 1

    ### remove egde
    erosion_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(5,5), anchor=(erosion_size, erosion_size)) # unsure value
    removed_edge_img = cv2.erode(img, erosion_element)

    ### Bright Pixel corecction
    hsv_img = cv2.cvtColor(removed_edge_img, cv2.COLOR_BGR2HSV_FULL)
    h_channel = hsv_img[:,:,0]
    # convert h_channel to floating point. max_h = 360
    h_channel = h_channel.astype(np.float16) / 360

    ### brightness correction
    r,h = h_channel.shape
    d = h_channel.reshape((-1)) # reshape to 1 dimensional list
    n = np.unique(d) # n is list of unique value of h_chanel from smallest to biggest
    
    ten_percent_largest_index = math.floor(n.size * 90 / 100)
    for i in range(0, d.size):
        if d[i] > n[ten_percent_largest_index]:
            d[i] = n[ten_percent_largest_index]
    
    plt.imshow(hsv_img[:,:,0], cmap='gray')
    
    h_channel = d.reshape((r,h)) # reconstruct channel

    ### adjust constrast
    
    h_adjusted = adjustConstrast(h_channel)
    
    
    for i in range(0,r):
        for y in range(0,h):
            if h_adjusted[i,y] == 0:
                h_adjusted[i,y] = 1

    ### Histogram construction
    num_of_bin = 100
    bin = divideIntoBucket(h_adjusted, num_of_bin)
    bin['1.00'] = 0
    # plt.bar(bin.keys(), bin.values(), color='g')
    ### Histogram Analysis
    B,V = max(bin.items(), key=operator.itemgetter(1))
    R = 0
    limit = 0

    key_list = list(bin.keys())

    index_of_B = Decimal(B) * num_of_bin
    if index_of_B <= 40:
        limit = Decimal(0.2) * V
    else:
        limit = Decimal(0.7) * V


    for key in reversed(bin):
        if bin[key] > limit:
            R = key
            break

    T = 2 * Decimal(R) / 3

    ### restruct image
    r, h = h_adjusted.shape
    for i in range(0, r):
        for y in range(0, h):
            if h_adjusted[i,y] > T:
                h_adjusted[i,y] = 1

    # now h_adjusted is a mask for disease

    for i in range(0, r):
        for y in range(0, h):
            if h_adjusted[i,y] == 1:
                img[i,y,:] = 0
                
    return img


def feature_extraction_whole_img_T(img, label):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    h = img[:,:,1]
    h2 = img[:,:,2]
    h3 = img[:,:,0]
    
    d = pd.DataFrame()
    
    ## color
    mean = np.mean(h)
    mean2 = np.mean(h2)
    mean3 = np.mean(h3)
    ## Texture

    grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    glcm = greycomatrix(grey_img, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)

    contrast = greycoprops(glcm, prop = 'contrast')
    dissimilarity = greycoprops(glcm, prop = 'dissimilarity')
    homogeneity = greycoprops(glcm, prop = 'homogeneity')
    asm = greycoprops(glcm, prop = 'ASM')
    energy = greycoprops(glcm, prop = 'energy')
    correlation = greycoprops(glcm, prop = 'correlation')
    
    # row
    d = d.append(pd.DataFrame({
        'color': [mean],
        'color2': [mean2],
        'light': [mean3],
        'contrast':[contrast],
        'dissimilarity':[dissimilarity],
        'homogeneity':[homogeneity],
        'asm':[asm],
        'energy':[energy],
        'correlation':[correlation],
        'label': [label]
    }))
    
    return d