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

def crop_quadrangle(image, vertices, width, height):
    """
    Crop a quadrangle from an image using perspective transformation.
    
    Parameters:
    - image: Input image (numpy array).
    - vertices: List of four vertex coordinates [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] representing the corners of the quadrangle.
    - width: Width of the output cropped quadrangle.
    - height: Height of the output cropped quadrangle.

    Returns:
    - Cropped quadrangle region as a numpy array.
    """
    dst_vertices = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    if len(vertices) != 4 or len(dst_vertices) != 4:
        raise ValueError("Both vertices and dst_vertices must have 4 points each.")
    vertices = np.array(vertices, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(vertices, dst_vertices)
    cropped_quadrangle = cv2.warpPerspective(image, matrix, (width, height))
    return cropped_quadrangle

def correct_skew(image, delta=0.2, limit=5, type=1):
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

def rotate_an_image(image, best_angle):
    """
    Given an input image and the best rotation angle calculated by correct_skew, 
    this function rotates the image to straighten it.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def rotate_list_iamges(list_images):
    """
    This function applies skew correction and rotation to a list of images. 
    It takes a list of images, performs skew correction, and returns a list of straightened images.
    """
    results_rotate: list = []
    for i in range(len(list_images)):
        cropped_quadrangle = cv2.cvtColor(list_images[i], cv2.COLOR_RGB2GRAY)
        best_angle = correct_skew(cropped_quadrangle, type=1)
        cropped_quadrangle = rotate_an_image(cropped_quadrangle, best_angle)
        results_rotate.append(cropped_quadrangle)
    return results_rotate


def streight(image_A):
    """
    This function applies a perspective transformation 
    to an image to straighten it based on predefined points.
    """
    mask = np.zeros((1400, 1600, 3))
    mask[320:,:1440] = image_A
    pts1 = np.float32([[130, 450], [1305, 250],
                [250, 1390], [1424, 1168]])
    pts2 = np.float32([[0, 0], [1280, 0],
                    [0, 1024], [1280, 1024]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    image_B = cv2.warpPerspective(mask, matrix, (1280, 1024))
    image_B = image_B.astype(np.uint8)
    return image_B



def crop_items_ocr(image_B):
    """
    This function crops items of interest (regions of text) 
    from an input image using perspective transformation.
    """
    list_images_crop: list =[]
    for i in range(len(names)):
        (w, h) = coordinates[i][6]
        cropped_image = crop_quadrangle(image_B, coordinates[i][:4], w, h)
        list_images_crop.append(cropped_image)
    
    return list_images_crop


def OCR_Reader(images_rotate):
    """
    This function performs OCR (optical character recognition) 
    on a list of rotated images and extracts text information from them.
    """
    results: dict = {}
    num_image_crop = len(images_rotate)
    for i in range(num_image_crop):
        result = ocr.ocr(img=images_rotate[i], cls=True)
        result = result[0]
        
        try:
            #boxes = [line[0] for line in result]
            txts = [(((line[1][0].replace('O', '0')).replace(' ', '')).replace(' ', '')).replace('-','') for line in result]
            #scores = [line[1][1] for line in result]
            for k in range(len(txts)):
                txts[k] = ''.join(filter(lambda char: char.isdigit() or char == '.' or char == '-', txts[k])) 
            
            results[names[i]] = txts
        except:
            results[names[i]] = ' '
    return results

def process(frame):
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
    t1 = time.time()
    #Step 1: streight_image
    frame = streight(frame)

    #Step 2: crop_item ocr
    list_image_crop = crop_items_ocr(frame)

    #Step 3.1: rotate list_image_crop
    results_image_rotate = rotate_list_iamges(list_image_crop)
    print(time.time() - t1)
    #Step 3.2: OCR Reader
    results = OCR_Reader(results_image_rotate)

    return results

if __name__ == "__main__":
    device = torch.device(f"cuda:0" if torch.cuda.is_available() and not False else "cpu")
    use_gpu = False
    if device == "cuda:0":
        use_gpu = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, 
        default="./data/Sample_PMS.mp4",
        help="Path to input video file"
    )
    parser.add_argument(
        "--num_frame", type=int, 
        default=15,
        help="After how many frames, it will be processed once"
    )
    parser.add_argument(
        "--maps", type=str,
        default= "./data/map_detect.txt",
        help="Path to map_detect file"
    )

    args = parser.parse_args()
    
    # data
    maps = args.maps
    in_video_path = args.input

    #Get maps2crop table
    coordinates: list = []
    names: list = []
    with open(maps, 'r') as file:
        for line in file:
            # Split each line by tab ('\t') and strip any leading/trailing whitespace
            values = line.strip().split('\t')
            line_tuple = [
                (int(values[i]), int(values[i+1])) for i in range(1, 14, 2) 
            ]
            names.append(values[0])
            coordinates.append(line_tuple)

    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)

    cap = cv2.VideoCapture(in_video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
        
    frame_number = 0 
    capture_interval = args.num_frame
    while True:
            ret, frame = cap.read()
            if not ret:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if frame_number % capture_interval == 0:
                start = time.time()
                results_OCR = process(frame=frame)
                end = time.time()
                print('Time: ', end - start, '\n', results_OCR)
    cap.release()
    
            