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


def crop_quadrangle(image, vertices, width, height):
    """
    Crop a quadrangle from an image using perspective transformation.
    :param image: Input image (numpy array).
    :param vertices: List of four vertex coordinates [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    :return: Cropped quadrangle region.
    """
    dst_vertices = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    if len(vertices) != 4 or len(dst_vertices) != 4:
        raise ValueError("Both vertices and dst_vertices must have 4 points each.")
    vertices = np.array(vertices, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(vertices, dst_vertices)
    cropped_quadrangle = cv2.warpPerspective(image, matrix, (width, height))
    cropped_quadrangle = cv2.cvtColor(cropped_quadrangle, cv2.COLOR_RGB2GRAY)
    best_angle = correct_skew(cropped_quadrangle, type=1)
    cropped_quadrangle = rotate(cropped_quadrangle, best_angle)
    return cropped_quadrangle

def correct_skew(image, delta=0.2, limit=5, type=1):
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


def streight(a_frame):
    pts1 = np.float32([[130, 450], [1305, 250],
                [250, 1390], [1424, 1168]])
    pts2 = np.float32([[0, 0], [1280, 0],
                    [0, 1024], [1280, 1024]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(a_frame, matrix, (1280, 1024))
    return result


def processing(input_path, output_path, coordinates, ocr):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
    output_video = cv2.VideoWriter(output_path, fourcc, 30, (1440, 1080))
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    frame_number = 0 
    capture_interval = 1
    mask = np.zeros((1400, 1600, 3))
    with open('./data/labels.txt', 'w') as f:
        num_error = 0
        while True:
            ret, frame = cap.read()
            mask[320:,:1440] = frame
            if not ret:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if frame_number % capture_interval == 0:
                image_filename = f'frame_{frame_number:04d}.png'
                new_frame = streight(mask)
                new_frame = new_frame.astype(np.uint8)
                #cv2.imwrite('./new_frame.png', new_frame)
                f.write(image_filename + "\t")
                
                for i in range(9):
                    #crop table2subimage
                    (w, h) = coordinates[i][6]
                    cropped_image = crop_quadrangle(new_frame, coordinates[i][:4], w, h)
                    cv2.imwrite(f'./sample/crop{i}.png', cropped_image)
                    result = ocr.ocr(cropped_image, cls=True)
                    result = result[0]

                    x1, y1 = coordinates[i][4]  # Top-left corner
                    x2, y2 = coordinates[i][5]  # Bottom-right corner
                    white = (255, 255, 255)
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), white, thickness=cv2.FILLED)
                    
                    try: 
                        boxes = [line[0] for line in result]
                        txts = [(((line[1][0].replace('O', '0')).replace(' ', '')).replace('.', '')).replace('-','') for line in result]
                        scores = [line[1][1] for line in result]
                        
                        for k in range(len(txts)):
                            txts[k] = ''.join(filter(lambda char: char.isdigit() or char == '.' or char == '-', txts[k]))
                            frame = cv2.putText(frame, txts[k], (x1+10, y1+20), font, font_scale, font_color, thickness, cv2.LINE_AA)
                            y1 += 20
                            f.write(txts[k]+'\t')
                        #cv2.putText(frame, '-----------', (5, y-15), font, font_scale, font_color, thickness, cv2.LINE_AA)
                    except:
                        print('error')
                        num_error += 1
                        cv2.imwrite(f"./error/{i}_error{num_error}.png", cropped_image)
                        cv2.putText(frame, txts[k], (x1+10, y1+20), font, font_scale, font_color, thickness, cv2.LINE_AA)
                    
                f.write("\n")
            frame_number += 1
            output_video.write(frame)
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
        default="./data/Sample_PMS.mp4",
        help="Path to input video file"
    )
    parser.add_argument(
        "--output", type=str,
        default="./data/output_video.avi",
        help="Path to output video file"
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
    out_video_path = args.output

    #Get maps2crop table
    coordinates: list = []
    with open(maps, 'r') as file:
        for line in file:
            # Split each line by tab ('\t') and strip any leading/trailing whitespace
            values = line.strip().split('\t')
            line_tuple = [
                (int(values[i]), int(values[i+1])) for i in range(1, 14, 2) 
            ]
            coordinates.append(line_tuple)
    
    #Custom PutText 2 Outvideo
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = .55
    font_color = (0, 255, 0)
    thickness = 2

    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)
    processing(
        input_path=in_video_path, 
        output_path=out_video_path,
        coordinates=coordinates,
        ocr=ocr
    )

    
            