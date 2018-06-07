#! -*- coding:utf-8 -*-

import time
import zipfile
from collections import defaultdict
from scipy.stats import itemfreq
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import feature
from PIL import Image as IMG
import numpy as np
import pandas as pd 
import operator
import cv2
import os 
import io
import gc

# add multiprocessing library
import multiprocessing
from multiprocessing import Pool

def parallelize_dataframe(df, func):
	# Set Multiprocessing Function
	num_partitions = 4
	num_cores = multiprocessing.cpu_count()

	#a,b = np.array_split(df, num_partitions)
	a,b,c,d = np.array_split(df, num_partitions)
	pool = Pool(num_cores)
	#df = pd.concat(pool.map(func, [a,b]))
	df = pd.concat(pool.map(func, [a,b,c,d]))
	pool.close()
	pool.join()
	return df

# Set scoring B&W Function
def color_analysis(img):
	# obtain the color palatte of the image 
	palatte = defaultdict(int)
	for pixel in img.getdata():
		palatte[pixel] += 1
	
	# sort the colors present in the image 
	sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
	light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
	for i, x in enumerate(sorted_x[:pixel_limit]):
		if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
			dark_shade += x[1]
		if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
			light_shade += x[1]
		shade_count += x[1]
		
	light_percent = round((float(light_shade)/shade_count)*100, 2)
	dark_percent = round((float(dark_shade)/shade_count)*100, 2)
	return light_percent, dark_percent

def perform_color_analysis(img, flag):
	path = images_path + img 
	#path = io.BytesIO(z.read(img))
	
	try:
		im = IMG.open(path) #.convert("RGB")
		
		# cut the images into two halves as complete average may give bias results
		size = im.size
		halves = (size[0]/2, size[1]/2)
		im1 = im.crop((0, 0, size[0], halves[1]))
		im2 = im.crop((0, halves[1], size[0], size[1]))
	except Exception as e:
		return -999
		
	try:
		light_percent1, dark_percent1 = color_analysis(im1)
		light_percent2, dark_percent2 = color_analysis(im2)
	except Exception as e:
		return -999

	light_percent = (light_percent1 + light_percent2)/2 
	dark_percent = (dark_percent1 + dark_percent2)/2 
	
	#return pd.Series([dark_percent, light_percent])
	if flag == 'black':
		return dark_percent
	elif flag == 'white':
		return light_percent
	else:
		return None

# Set scoring APW Function
def average_pixel_width(img):
	path = images_path + img 
	try:
		im = IMG.open(path)    
		im_array = np.asarray(im.convert(mode='L'))
		edges_sigma1 = feature.canny(im_array, sigma=3)
		apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
		return apw*100
	except Exception as e:
		return -999

# Set scoring DominantColor Function
def get_dominant_color(img):
	path = images_path + img
	try:
		img = cv2.imread(path)
		arr = np.float32(img)
		pixels = arr.reshape((-1, 3))

		n_colors = 5
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
		flags = cv2.KMEANS_RANDOM_CENTERS
		_, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

		palette = np.uint8(centroids)
		quantized = palette[labels.flatten()]
		quantized = quantized.reshape(img.shape)

		dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
		return dominant_color
	
	except Exception as e:
		return [-999, -999, -999]

# Set scoring AverageColor Function
def get_average_color(img):
	path = images_path + img 
	
	try:
		img = cv2.imread(path)
		average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
		return average_color
	
	except Exception as e:
		return [-999, -999, -999]

# Set scoring Dimensions Fucntion
def getSize(filename):
	filename = images_path + filename
	try:
		st = os.stat(filename)
		return st.st_size
	except Exception as e:
		return -999
	
def getDimensions(filename):
	filename = images_path + filename
	try:
		img_size = IMG.open(filename).size
		return img_size
	except Exception as e:
		return (-999,-999)

# Set scoring Blurness Function
def get_blurrness_score(image):
	path = images_path + image 
	try:
		image = cv2.imread(path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		fm = cv2.Laplacian(image, cv2.CV_64F).var()
		return fm
	except Exception as e:
		return -999

images_path = "input/"
imgs = os.listdir(images_path)

print("Size of images : ", len(imgs))

features = pd.DataFrame()
features['image'] = imgs

# For test
features = features[154461:154661]

print("Size of features : ", len(features))

# Set Multiprocessing Function
#num_partitions = 4
#num_cores = multiprocessing.cpu_count()

# Scoring dullness and whilteness
print("Starting score dullness and whiteness..")
stime = time.time()
	
fe_bw = features.copy()
def score_bw(data):
	data["dullness"] = data["image"].apply(lambda x : perform_color_analysis(x, "black"))
	data["whiteness"] = data["image"].apply(lambda x : perform_color_analysis(x, "white"))
	return data

fe_bw = parallelize_dataframe(fe_bw, score_bw)
fe_bw.to_pickle("fe_bw.df")	

etime = time.time() - stime

print(time.ctime())
print("time is ", etime)

del fe_bw
gc.collect()

# Scoring average pixel width
print("Starting score averag pixcel width..")
stime = time.time()

fe_apw = features.copy()
def score_apw(data):
	data["average_pixel_width"] = data["image"].apply(average_pixel_width)
	return data

fe_apw = parallelize_dataframe(fe_apw, score_apw)
fe_apw.to_pickle("fe_apw.df")

etime = time.time() - stime

print(time.ctime())
print("time is ", etime)

del fe_apw
gc.collect()
	

# Scoring dominant color

# Scoring average color
print("Starting score average color..")
stime = time.time()
	
fe_ac = features.copy()
def score_ac(data):
	data["average_color"] = data["image"].apply(get_average_color)
	return data

fe_ac = parallelize_dataframe(fe_ac, score_ac)
fe_ac.to_pickle("fe_ac.df")

etime = time.time() - stime

print(time.ctime())
print("time is ", etime)

del fe_ac
gc.collect()	

# Scoring dimensions
print("Starting score dimensions..")
stime = time.time()
	
fe_dim = features.copy()
def score_dim(data):
	data["item_size"] = data["image"].apply(getSize)
	data["temp_size"] = data["image"].apply(getDimensions)
	return data

fe_dim = parallelize_dataframe(fe_dim, score_dim)
fe_dim.to_pickle("fe_dim.df")

etime = time.time() - stime

print(time.ctime())
print("time is ", etime)
	
del fe_dim
gc.collect()
	
# Scoring blurness
print("Starting score blurness..")
stime = time.time()

fe_bl = features.copy()
def score_bl(data):
	data["blurness"] = data["image"].apply(get_blurrness_score)
	return data

fe_bl = parallelize_dataframe(fe_bl, score_bl)
fe_bl.to_pickle("fe_bl.df")

etime = time.time() - stime

print(time.ctime())
print("time is ", etime)

del fe_bl
gc.collect()

fe_bw = pd.read_pickle("fe_bw.df")
fe_ac = pd.read_pickle("fe_ac.df")
fe_apw = pd.read_pickle("fe_apw.df")
fe_dim = pd.read_pickle("fe_dim.df")
fe_bl = pd.read_pickle("fe_bl.df")

fe_ac["avg_r"] = fe_ac["average_color"].apply(lambda x: x[0]) / 255
fe_ac["avg_g"] = fe_ac["average_color"].apply(lambda x: x[1]) / 255
fe_ac["avg_b"] = fe_ac["average_color"].apply(lambda x: x[2]) / 255

fe_dim["width"] = fe_dim["temp_size"].apply(lambda x : x[0])
fe_dim["height"] = fe_dim["temp_size"].apply(lambda x : x[1])

fe = fe_bw.merge(fe_ac, on="image", how="left")
fe = fe.merge(fe_apw, on="image", how="left")
fe = fe.merge(fe_dim, on="image", how="left")
fe = fe.merge(fe_bl, on="image", how="left")

fe.drop(["average_color", "temp_size"], 1, inplace=True)

fe.to_pickle("features.df")
