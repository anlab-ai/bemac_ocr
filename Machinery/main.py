import cv2
import numpy as np
import numpy as np
import cv2
from scipy.ndimage import interpolation as inter
from paddleocr import PaddleOCR,draw_ocr
import logging
import os
import re
import time
import torch
import argparse
def crop_quadrangle(image, vertices, width, height):
    """
    Crop a quadrangle from an image using perspective transformation.
    :param image: Input image (numpy array).
    :param vertices: List of four vertex coordinates [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    :return: Cropped quadrangle region.
    """
    dst_vertices = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    vertices = np.array(vertices, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(vertices, dst_vertices)
    cropped_quadrangle = cv2.warpPerspective(image, matrix, (width, height))
    #cropped_quadrangle = cv2.cvtColor(cropped_quadrangle, cv2.COLOR_RGB2GRAY)
    #best_angle = correct_skew(cropped_quadrangle, type=1)
   # cropped_quadrangle = rotate(cropped_quadrangle, best_angle)
    return cropped_quadrangle

def correct_skew(image, delta=1, limit=5, type=1):
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

def rotate(image, best_angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def streight(a_frame, pts1, pts2, w, h):
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(a_frame, matrix, (w, h))
    return result

def crop(img):
    """
    img: gray img
    """
    _,bi_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # bi_img = cv2.dilate(bi_img, (7,7))
    contours,_ = cv2.findContours(bi_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_bbox = []
    imgh, imgw  = bi_img.shape
    for contour in contours:
        if cv2.contourArea(contour)< 27:
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
    new_img = cv2.copyMakeBorder(crop_img, 72, 72, 0, 0, cv2.BORDER_CONSTANT,value=background_value)

    return new_img


def processing(input_path, output_path, coordinates, ocr):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
    output_video = cv2.VideoWriter(output_path, fourcc, 15, (1280, 720))
    pts1 = np.float32([[240, 30], [1165, 0],
                [145, 718], [1279, 705]])
    pts2 = np.float32([[0, 0], [1280, 0],
                    [0, 720], [1280, 720]])
    s_w = 1280
    s_h = 720
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    frame_number = 0 
    capture_interval = 5
    with open('./data/labels.txt', 'w') as f:
        num_error = 0
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            mean_time = []
            if frame_number % capture_interval == 0:
                image_filename = f'frame_{frame_number:04d}.png'
                white = (255, 255, 255)
                start_time = time.time()
                new_frame = streight(frame, pts1, pts2, s_w, s_h)
                for i in range(24):
                    (w, h) = coordinates[i][6]
                    cropped_image = crop_quadrangle(new_frame, coordinates[i][:4], w, h)
                    if i == 1 or i == 4:
                        cropped_image = crop(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY))
                    #cv2.imwrite(f'./sample/{i}.png', cropped_image)
                    x1, y1 = coordinates[i][4]  # Top-left corner
                    x2, y2 = coordinates[i][5]  # Bottom-right corner
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), white, thickness=cv2.FILLED)
                    result = ocr.ocr(cropped_image, cls=False, inv=False)
                    result = result[0]
                    try: 
                        #boxes = [line[0] for line in result]
                        txts = [((((line[1][0].replace('o', '0')).replace(' ', '')).replace('O', '0'))).replace('°', '0') for line in result]
                        #scores = [line[1][1] for line in result]
                        if i != 16:
                            txts[-1] = ''.join(filter(lambda char: char.isdigit() or char == '.' or char == '-', txts[-1]))
                            frame = cv2.putText(frame, txts[-1], (x1+10, y1+15), font, font_scale, font_color, thickness, cv2.LINE_AA)
                        else:
                            txts[-2] = ''.join(filter(lambda char: char.isdigit() or char == '.', txts[-2]))
                            txts[-1] = ''.join(filter(lambda char: char.isdigit() or char == '.', txts[-1]))
                            
                            #Handle the case of misreading the Celsius symbol
                            #if txts[-1][-1] == '0':
                            #    txts[-1] = txts[-1][:-1]

                            
                            frame = cv2.putText(frame, txts[-2], (x1+10, y1+15), font, font_scale, font_color, thickness, cv2.LINE_AA)       
                            frame = cv2.putText(frame, txts[-1], (x1+10, y1+35), font, font_scale, font_color, thickness, cv2.LINE_AA)            
                    except:
                        print('error')
                        num_error += 1
                        cv2.imwrite(f"./error/error{num_error}_{frame_number}.png", cropped_image)
                end_time = time.time()
                print(f'Frame {frame_number}, time: ', end_time - start_time)
                output_video.write(frame)
                #cv2.imwrite('result/putText'+ image_filename, frame)
            frame_number += 1
        
    print("Num error: ", num_error)
    output_video.release()
    cap.release()

if __name__ == "__main__":
    device = torch.device(f"cuda:0" if torch.cuda.is_available() and not False else "cpu")
    use_gpu = False
    if device == "cuda:0":
        use_gpu = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, 
        default="./data/Sample_Machinery.mp4",
        help="Path to input video file"
    )
    parser.add_argument(
        "--output", type=str,
        default="./data/output_video.avi",
        help="Path to output video file"
    )
    parser.add_argument(
        "--maps", type=str,
        default= "./data/maps_detect.txt",
        help="Path to map_detect file"
    )
    args = parser.parse_args()
    
    # data
    maps = args.maps
    in_video_path = args.input
    out_video_path = args.output


    coordinates: list = []
    with open(maps, 'r') as file:
        for line in file:
            # Split each line by tab ('\t') and strip any leading/trailing whitespace
            values = line.strip().split('\t')
            line_tuple = [
                (int(values[i]), int(values[i+1])) for i in range(1, 14, 2) 
            ]
            coordinates.append(line_tuple)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = .53
    font_color = (0, 255, 0)
    thickness = 2

    #ocr = PaddleOCR()
    #custom_dict_file = "./data/custom_dict.txt"
    
    ocr = PaddleOCR(use_angle_cls=True, lang= 'en', use_gpu=use_gpu)

    processing(
        input_path=in_video_path, 
        output_path=out_video_path,
        coordinates=coordinates,
        ocr=ocr
        )
    
    
            