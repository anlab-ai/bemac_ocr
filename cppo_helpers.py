import cv2
import json
import numpy as np
from matching import PlanarMatching
from paddleocr import PaddleOCR


import os
import cv2
import numpy

def crop_image(img):
	cropped_images = []
	rectangles = []
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	m, stds  = cv2.meanStdDev(hsv)
	# print("mean " , m, stds)
	min_val = max(int(0.85*m[2]) , 25)
	lower_green = np.array([30,25,min_val])
	upper_green = np.array([100,255,255])
	binary_image = cv2.inRange(hsv, lower_green, upper_green)
	# cv2.imshow("binary " , binary_image)
	contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	area_min = 70  
	for contour in contours:
		area = cv2.contourArea(contour)
		if area >= area_min:
			x, y, w, h = cv2.boundingRect(contour)
			rectangles.append([x, y, w, h])
	try:
		rectangles, img = group_and_merge_rectangles(rectangles=rectangles, img=img)
	except:
		rectangles = []
	return img, rectangles


def group_and_merge_rectangles(rectangles, img):

	rectangles.sort(key=lambda r: r[0])
	distance_threshold = 20     
	clusters = []

	current_cluster = [rectangles[0]]

	for i in range(1, len(rectangles)):
		x1, y1, w1, h1 = rectangles[i]
		x2, _, w2, h2 = current_cluster[-1]
		distance_threshold = 0.95*max(h1,h2)

		distance = x1 - (x2 + w2)

		if distance <= distance_threshold:
			current_cluster.append(rectangles[i])
		else:
			clusters.append(current_cluster)
			current_cluster = [rectangles[i]]

	clusters.append(current_cluster)

	new_rectangles = []
	for cluster in clusters:
		
		x_min = min(rect[0] for rect in cluster)
		y_min = min(rect[1] for rect in cluster)
		x_max = max(rect[0] + rect[2] for rect in cluster)
		y_max = max(rect[1] + rect[3] for rect in cluster)
		new_rectangles.append([x_min, y_min, x_max, y_max])

	return new_rectangles, img


def warp_image(image):
	w = 220
	h = 80
	x = 40
	y = 5

	pts1 = np.float32([[65, 1], [257, 25],
					[224, 80], [48, 57]])
	pts2 = np.float32([[x, y], [x+w, y],
					[x+ w, y + h] , [x, y+ h]])
	matrix = cv2.getPerspectiveTransform(pts1, pts2)
	size_h , size_w = image.shape[:2]
	result = cv2.warpPerspective(image, matrix, (size_w, size_h))
	return result


def segment_device(image, input_size = 90):

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	_, thresh = cv2.threshold(gray, 30, 255,cv2.THRESH_BINARY)
	# define the kernel 
	kernel = np.ones((5, 5), np.uint8) 
	
	
	# erode the image 
	erosion = cv2.erode(thresh, kernel, 
						iterations=1) 
	
	# dilate the image 
	dilate = cv2.dilate(erosion, kernel, 
						iterations=3) 
	# find countour area max.

	contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	thresh_min = 100  
	index_max = -1
	area_max = 0
	boxes = []
	for i, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if area >= thresh_min:
			x, y, w, h = cv2.boundingRect(contour)
			boxes.append([x, y , x+ w , y + h])


	image_device = None
	res_detect = False
	if len(boxes) > 0 :
		
		x_min = int(min(rect[0] for rect in boxes))
		y_min = int(min(rect[1] for rect in boxes))
		x_max = int(max( rect[2] for rect in boxes))
		y_max = int(max( rect[3] for rect in boxes))
		res_detect = True
		image_device = image[ y_min: y_max , x_min:x_max]
		size_image = image_device.shape[:2]
		target_size = ( int(size_image[1]*input_size/size_image[0]) , input_size  )
		image_device = cv2.resize(image_device, target_size, interpolation=cv2.INTER_CUBIC)
		# image = cv2.rectangle(image, (x_min, y_min ),(x_max , y_max) , (0, 0 , 255), 2)
		image_device = warp_image (image_device)
	# cv2.imshow("image_device " , image_device)		
	return res_detect, image_device

		  
def crop_items( image, config_position):
		"""
		This function crops items of interest (regions of text) 
		from an input image using perspective transformation.
		"""
		h, w = image.shape[:2]
		list_images_crop = {}
		list_key_pose = {}
		for i in config_position.keys():
			list_images_crop[i] = None
			val = config_position[i]
			x = val["pose"][0]
			y = val["pose"][1]
			width = val["pose"][2]
			height = val["pose"][3]
			if x  < w/3:
				list_key_pose[2] = i
			else:
				list_key_pose[1] = i
		new_image, rectangles = crop_image(img=image)  
		pix_pad = 3
		if len(rectangles) > 0 :
			for i in range(len(rectangles)):
				x_min = int(max(0,  rectangles[i][0]-pix_pad))
				y_min = int(max(0, rectangles[i][1]-pix_pad))
				x_max = int(min(w, rectangles[i][2]+pix_pad))
				y_max = int(min(h, rectangles[i][3]+pix_pad))

				size_x = x_max - x_min
				size_y = y_max - y_min
				pix_pad_2 = 0
				if size_x < 0.6 * size_y:
					pix_pad_2 = int ((size_y - size_x)/2)
					x_min = int(max(0,  rectangles[i][0]-pix_pad_2))
					x_max = int(min(w, rectangles[i][2]+pix_pad_2))

				im = image[y_min:y_max, x_min:x_max]
				# cv2.imshow(f"im {i} " , im)
				if x_min > w/2 :
					list_images_crop[list_key_pose[1]] = im
				else:
					list_images_crop[list_key_pose[2]] = im

		return list_images_crop

def replace_char2digit(input):
	list_char = ["s","S", "z", "Z" , "p" , "P", "l", ":", "i", "I", "n", "o", "O", "D", "Q", "R", "B"]
	list_digit = ["5", "5" , "2", "2", "2" , "2", "1", "1" , "1","1", "0" , "0" , "0", "0" , "8", "8"]
	result = input
	for c , d in zip(list_char,list_digit):
		result = result.replace(c, d)
	return result

def correct_results(results):
	for  key in results.keys():
		val = results[key]
		f = val["val"]
		score = val["score"]
		str_trim = f.strip()
		pose = -1
		pose2 = -1
		if "." in str_trim:
			pose = str_trim.index(".")
		if "," in str_trim:
			pose2 = str_trim.index(",")

		is_minus = False
		if pose ==0:
			f = f.replace(".", "")
			is_minus = True
			
		if pose2 == 0 :
			f = f.replace(",", "")
			is_minus = True
		val["val"] = f
		if not(f.isdigit()) :
			if f in ["a", "n"]:
				val["val"] = 0
			else:
				val["val"] = "Nan"
				val["score"] = ""
		elif is_minus:
			val["val"] = f'-{val["val"]}'
		results[key] = val

	return results