import json
from tqdm import tqdm
from pathlib import Path
import os
import csv
import pandas as pd
import pickle
import time
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def read_data(file ):
	datas = {}
	with open(file) as file_obj: 
		reader_obj = csv.reader(file_obj) 
		for  row in reader_obj: 
			
			frame_count = row[0]
			num_data = len(row)//2
			data = []
			for i in range(num_data) :
				val = row[2*i +1].replace(f"'", "").strip() 
				val = val.replace('.',' .')
				score = row[2*i+2]
				list_val = val.replace('-',' -').split(' ')
				for v in list_val:
					data.append([v, score])
			datas[frame_count] = data
	# print(datas)
	# exit()
	return datas

def generate_word(datas):
	result_data = {}
	key_words = {}
	for key in datas:
		data  = datas[key]
		str = ''
		for i , r in enumerate(data):
			score = r[1]
			count = 1
			val = r[0].replace(f"'", "").strip()
			str = str + f' {val}'
			if  val in key_words.keys():
				score = max(key_words[val][0] , r[1])
				count = key_words[val][1] + 1
			key_words[val] = [score, count ] 
		
		result_data[key] = str.strip()

	print( key_words)

	return key_words, result_data

def detect_max_len(datas):

	frame_id_max = 0 
	matchs = {}
	for frame_count in datas.keys():
		str = datas[frame_count]
		score_sum = len(str)
		for i in range(len(str)):
			score_sum += float(str[i][1])
		
		matchs[frame_count] = score_sum
	# print("matchs " , matchs)
	sorted_by_scores = sorted(matchs.items(), key=lambda x:x[1], reverse=True)
	frame_id_max = sorted_by_scores[0][0]
	print("frame_id_max " , frame_id_max)
	return frame_id_max

def detect(result_data):

	frame_id_max = 0 
	matchs = {}
	for frame_count in result_data.keys():
		str = result_data[frame_count]
		score_sum = 0
		for frame_count2 in result_data.keys():
			str2 = result_data[frame_count2]
			score_sum += fuzz.ratio(str, str2)
		matchs[frame_count] = score_sum
	# print("matchs " , matchs)
	sorted_by_scores = sorted(matchs.items(), key=lambda x:x[1], reverse=True)
	frame_id_max = sorted_by_scores[0][0]
	print("frame_id_max " , frame_id_max)
	return frame_id_max

def correct_result (datas , key_words, frame_id_max, thresh = 70 , est = 0.01):
	result = datas[frame_id_max]
	# print("key_words " , key_words)
	list_match = list(key_words.keys())
	for i , r in enumerate(result):
		val_query = r[0] 
		out = process.extract(val_query, list_match, limit=10)
		res = {}
		for top, o in enumerate(out):
			val = o[0]
			s = float(o[1]) 
			diff_text = abs( len(val) - len(val_query))
			if s > thresh:
				s = float(key_words[val][0]) + float(est*key_words[val][1]) - diff_text*est*2
			else:
				s = 0
			res[val] = s
		sorted_by_scores = sorted(res.items(), key=lambda x:x[1], reverse=True)
		print("out ", val_query , out)
		max_val = sorted_by_scores[0]
		if  max_val[1] > float(r[1]) :
			result[i] = [max_val[0] , max_val[1]]
	text_out = ""
	for r in result:
		text_out = text_out + f' {r[0]}'
	text_out = text_out.strip()
	text_out = text_out.replace(' -','-').replace(' .','-') 
	return text_out



def read_path(base_folder ):
	paths = []
	for p, d, f in os.walk(base_folder):
		for file in f:
			if file.endswith('.csv') :
				# print("file" , file)
				paths.append(file)
	return paths

if __name__ == "__main__":
	base_folder_path = "output/"
	path_out = "result.txt"
	if os.path.exists(path_out):
		os.remove(path_out)
	paths = read_path(base_folder_path)
	for path in paths:
		# path = "IMG_0160.MOV.csv"
		print("=========================\n" , path)
		file = os.path.join(base_folder_path, path)
		datas = read_data(file)
		key_words, result_data = generate_word(datas)
		print("key_words " , key_words)
		# find text result 
		frame_id_max = detect_max_len( datas)
		# replace word from list text.
		result = correct_result (datas , key_words, frame_id_max)
		print("result " , result)
		with open(path_out, 'a') as f:
			f.write(f'{file}\t{result}\n')
		# exit()

