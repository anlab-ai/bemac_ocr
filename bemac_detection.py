import cv2
import numpy as np
import numpy as np
import cv2
from scipy.ndimage import interpolation as inter
from paddleocr import PaddleOCR
import os
import re
import torch
import argparse
import time
import sys
from pathlib import Path

from matching import PlanarMatching
import helpers
import cppo_helpers

device = torch.device(f"cuda:0" if torch.cuda.is_available() and not False else "cpu")


class BemacOCR():
	def __init__(self, type_divice=0):
		self.type_divice = type_divice
		name_device = ["machinery", "pms", "helicon", "cpporder"]
		
		
		use_gpu = False
		if device == "cuda:0":
			use_gpu = True
		self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)
		print("device:  ",name_device[type_divice])
		self.config_position , self.query_img = self.load_config̣(name_device[type_divice])
		self.size_template_width  = self.query_img.shape[1]
		self.size_template_hieght  = self.query_img.shape[0]
		self.M = PlanarMatching(self.query_img)

	def load_config̣(self, name_device):
		path_map = os.path.join(f"./config/{name_device}.xlsx")
		self.config_position = helpers.read_config_map(path_map)
		path_template = os.path.join(f"./template/{name_device}.jpg")
		self.query_img = cv2.imread(path_template)
		return self.config_position , self.query_img



	def crop(self, img, kk):
		"""
		img: gray img
		"""
		
		_,bi_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		# bi_img = cv2.dilate(bi_img, (7,7))
		contours,_ = cv2.findContours(bi_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contour_bbox = []
		imgh, imgw  = bi_img.shape
		for contour in contours:
			
			if kk != 9 and cv2.contourArea(contour)< 27:
	
				continue
			if kk == 9 and cv2.contourArea(contour)< 60:
				continue
			rect = cv2.boundingRect(contour)
			x,y,w,h = rect

			x1, y1, x2, y2 = x, y, x+w, y+h
			if y1 >imgh*0.6 and y2 > imgh*0.8 :
				continue
			if h<=6 and (w>=10 or w<=6):
				continue
			contour_bbox.append([x1,y1,x2,y2])
		if (len(contour_bbox)==0):
			print("Not Found contour")
			return img

		crop_imgs = []
		contour_bbox = np.array(contour_bbox)
		fx1= np.min(contour_bbox[:, 0])
		fy1= np.min(contour_bbox[:, 1])
		fx2= np.max(contour_bbox[:, 2])
		fy2= np.max(contour_bbox[:, 3])
		pad = [2,2,2,2]
		fx1 = fx1 - pad[0] if fx1 - pad[0] >= 0 else 0
		fy1 = fy1 - pad[1] if fy1 - pad[1] >= 0 else 0
		fx2 = fx2 + pad[2] if fx2 + pad[2] < imgw else imgw-1
		fy2 = fy2 + pad[3] if fy2 + pad[3] < imgh else imgh-1
		crop_img = img[fy1:fy2, fx1:fx2]

		crop_img = cv2.resize(crop_img, (48, 48),
				interpolation = cv2.INTER_CUBIC)
		background_value = int(crop_img[0,0]/2)
		#new_img = cv2.copyMakeBorder(crop_img, 72, 72, 0, 0, cv2.BORDER_CONSTANT,value=background_value)
		return crop_img# new_img


	def correct_skew(self, image, delta=0.2, limit=5, type=1):
		"""
		This function corrects the skew in an input image. 
		It calculates the best rotation angle to straighten the image
		based on the content's horizontal or vertical distribution.
		
		"""
		
		def determine_score(arr, angle, type):
			data = inter.rotate(arr, angle, reshape=False, order=0)
			histogram = np.sum(data, axis=type, dtype=float)
			arr_his = np.sum(arr, axis=type, dtype=float)
			score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
			return histogram, score

		thresh = image
		scores = []
		angles = np.arange(-limit, limit + delta, delta)
		for angle in angles:
			histogram, score = determine_score(thresh, angle, type)
			scores.append(score)
		best_angle = angles[scores.index(max(scores))]
		return best_angle

	def rotate_an_image(self,image, best_angle):
		"""
		Given an input image and the best rotation angle calculated by correct_skew, 
		this function rotates the image to straighten it.
		"""
		(h, w) = image.shape[:2]
		center = (w // 2, h // 2)
		M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
		rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
		return rotated

	def rotate_list_iamges(self, list_images):
		"""
		This function applies skew correction and rotation to a list of images. 
		It takes a list of images, performs skew correction, and returns a list of straightened images.
		"""
		results_rotate = {}
		for i in list_images.keys():
			image = list_images[i]
			cropped_quadrangle = cv2.cvtColor(image ,cv2.COLOR_RGB2GRAY)
			best_angle = self.correct_skew(cropped_quadrangle, type=1)
			cropped_quadrangle = self.rotate_an_image(cropped_quadrangle, best_angle)
			results_rotate [i] = cropped_quadrangle
		return results_rotate

	def streight_sift_matcḥ̣̣(self, img):

		res, area, H = self.M.is_image_relevant(img_2=img,output_vis=False)
		if res:
			img_streight = cv2.warpPerspective(img, H, (self.size_template_width, self.size_template_hieght))
			img_streight = img_streight.astype(np.uint8)

		return res , img_streight


	def crop_items_ocr(self, image_B):
		"""
		This function crops items of interest (regions of text) 
		from an input image using perspective transformation.
		"""
		list_images_crop = {}
		for i in self.config_position.keys():
			val = self.config_position[i]
			x = val["pose"][0]
			y = val["pose"][1]
			width = val["pose"][2]
			height = val["pose"][3]
			cropped_image = image_B[y:y+height, x:x+width]
			if self.type_divice == 0 and  (int(i) == 1 or int(i) == 4 or int(i) == 9):
				cropped_image = self.crop(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY), i)
				cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
			list_images_crop[i] = cropped_image
		
		return list_images_crop


	def OCR_Reader(self,images_rotate, frame_number):
		"""
		This function performs OCR (optical character recognition) 
		on a list of rotated images and extracts text information from them.
		"""
		results: dict = {}
		
		num_image_crop = len(images_rotate)
		for i in images_rotate.keys():
			image = images_rotate[i]
			val = {}
			if image is None:
				val["val"] = "Nan"
				val["score"] = ""
				results[i] = val
				continue

			if self.type_divice == 0 and  (int(i) == 1 or int(i) == 4 or int(i) == 9):
				result = self.paddle_ocr.ocr(img=image, cls=True, inv=True, bin= False, det=False, rec=False)
			elif self.type_divice == 1 and  (int(i) in [1,2,6,8,9]):
				result = self.paddle_ocr.ocr(img=image, cls=True, inv=True, bin= False, det=False, rec=False)
			elif self.type_divice == 3 :
				result = self.paddle_ocr.ocr(img=image, cls=True, inv=False, bin= False, det=False, rec=False)
			else:
				result = self.paddle_ocr.ocr(img=image, cls=False)

			
			result = result[0]
			
			# print("result" , result)
			# cv2.imwrite("aaa.jpg" , image)
			# print("result" , result)
			# if result is None :
			#     continue
			# exit()
			
			try:
				#boxes = [line[0] for line in result]
				scores = [line[-1][1] for line in result]
				txts = [(((line[-1][0].replace('O', '0')).replace(' ', '')).replace(' ', '')) for line in result]
				print(txts)
				for k in range(len(txts)):
					txts[k] = ''.join(filter(lambda char: char.isdigit() or char == '.', txts[k]))

				if self.type_divice == 1:
					txts = [line.replace('-','') for line in  txts]
				
				info = ""
				if len(txts) > 0 :
					info = txts[0]
				if len(info) > 0 :
					val["val"] = info
					val["score"] = scores[0]
				else:
					#cv2.imwrite(f'./error/{frame_number}_{i}.png', images_rotate[i])
					val["val"] = "Nan"
					val["score"] = ""
				
			
			except:
				#cv2.imwrite(f'./error/{frame_number}_{i}.png', images_rotate[i])
				val["val"] = "Nan"
				val["score"] = ""
			
			results[i] = val
		return results


	def process(self, frame, frame_number):
		"""
		This is the main processing function that ties everything together. 
		It takes a frame (image) as input, performs the following steps:

		Step 1: Straightens the frame using streight.
		Step 2: Crops items of interest using crop_items_ocr.
		Step 3:
			+) Rotates the cropped items using rotate_list_images.
			+) Performs OCR on the rotated images using OCR_Reader.

		The results are returned as a dictionary, where each item corresponds to a named region of interest.
		"""
		results: dict = {}
		results_scores: dict = {}

		if (self.type_divice < 3):
			t1 = time.time()
			#Step 1: streight_image

			res, frame = self.streight_sift_matcḥ̣̣(frame)
			print( "time warp " , time.time() - t1)
			t1 = time.time()

			if not(res):
				return results

		
			#Step 2: crop_item ocr
			list_image_crop = self.crop_items_ocr(frame)
			print( "time crop " , time.time() - t1)
			t1 = time.time()
			#Step 3.1: rotate list_image_crop
			results_image_rotate = self.rotate_list_iamges(list_image_crop)
			print( "time pre process " , time.time() - t1)
		else:
			# segment image.
			res, img_device = cppo_helpers.segment_device(frame)
			if not(res):
				return results

			#Step 2, 3.1: crop_item ocr .
			results_image_rotate = cppo_helpers.crop_items(img_device, self.config_position)
		t1 = time.time()

		#Step 3.2: OCR Reader
		results = self.OCR_Reader(results_image_rotate, frame_number)
		print( "time ocr " , time.time() - t1)
		t1 = time.time()

		if self.type_divice == 3:
			results = cppo_helpers.correct_results(results)

		return results, frame

	def process_video(self,video_path, path_out ):
		if os.path.exists(path_out):
			os.remove(path_out)
		
		helpers.write_title(path_out,  self.config_position)
		
		cap = cv2.VideoCapture(video_path)

		if not cap.isOpened():
			print("Error: Could not open video file.")
			exit()
		fps = int(cap.get(cv2.CAP_PROP_FPS) + 0.1) 
		print("fps ",fps)
		frame_number = 0
		count = 0
		while True:
				ret, frame = cap.read()
				if not ret:
					break
				print("frame_number ", frame_number)
				if frame_number % fps == 0:
					start = time.time()

					results, frame = self.process(frame=frame, frame_number= frame_number)
					# cv2.imwrite("frame.jpg", frame)
					# print("results ", results)
					# exit()
					str_data = helpers.convert_data_raw(path_out ,results, self.config_position )
					time_detect= helpers.convert_time(count)
					str_data = f'{count}, {time_detect} {str_data}'
					# print("str_data" , str_data)
					count += 1 
					with open(path_out, 'a') as file:
						file.write(str_data + "\n")
				frame_number += 1
				# if count > 3:
				# 	exit()
		cap.release()

	def drawText(self, image, results):
		white = (255, 255, 255)
		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = .53
		font_color = (0, 255, 0)
		thickness = 2

		for i in self.config_position.keys():
			val = self.config_position[i]
			x = val["pose"][0]
			y = val["pose"][1]
			w = val["pose"][2]
			h = val["pose"][3]
			name = val["name"]
			info = results[i]["val"]
			image = cv2.putText(image, f'{name}:{results[i]["val"]}', (x-5, y+h-5), font, font_scale, font_color, thickness, cv2.LINE_AA)
		return image

def read_path(base_folder ):
	paths = []
	for p, d, f in os.walk(base_folder):
		for file in f:
			if file.endswith('.mp4') or file.endswith('.MP4'):
				# print("file" , file)
				paths.append(file)

	return paths

if __name__ == "__main__":
	
	
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--input", type=str, 

		default="/media/anlab/800gb/bemac/Machinery",
		help="Path to input video file"
	)

	parser.add_argument(
		"--output", type=str, 

		default="results",
		help="Path to input video file"
	)

	parser.add_argument(
		"--type", type=str,
		default= 0,
		help="Path to map_detect file"
	)

	args = parser.parse_args()
	type = int(args.type)
	seg_model = BemacOCR(type_divice = type)

	base_folder_path = args.input
	base_save_folder=  args.output
	Path(base_save_folder).mkdir(parents=True, exist_ok=True)
	
	id_folder = base_folder_path.split("/")[-1]
	sub_path_save = os.path.join(base_save_folder, id_folder)
	Path(sub_path_save).mkdir(parents=True, exist_ok=True)
	print(sub_path_save, id_folder)
	paths = read_path(base_folder_path)
	print("list video " ,paths )
	for file in paths:
		base = os.path.basename(file)
		path_out =  os.path.join(sub_path_save, f'{base}.csv')
		in_video_path = os.path.join(base_folder_path, file)
		seg_model.process_video(in_video_path , path_out)
	
			