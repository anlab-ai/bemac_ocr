import cv2
import numpy as np
import os
import csv
from argparse import ArgumentParser
from PIL import Image, ImageFilter
from skimage.filters import unsharp_mask
from paddleocr import PaddleOCR
import random
import time



ocr = PaddleOCR(lang='en')


def unsharpmask(im, type = 1):
    if type == 0:
        im1 = np.copy(im).astype(float)
        for i in range(3):
            im1[...,i] = unsharp_mask(im[...,i], radius=5, amount=6)
        return im1
    elif type == 1:
        im = Image.fromarray(im)
        im2 = im.filter(ImageFilter.UnsharpMask(radius=9, percent=400))
        #print(im2)
        return np.array(im2)/255
    return None


def process_video(video_path):
    file_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    folder_path = './output'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    frame_num = 0
    fps = 10 #int(cap.get(cv2.CAP_PROP_FPS) + 0.1)//3
    with open(os.path.join(folder_path, file_name + '.csv'), 'w', newline= '') as file:
        writer = csv.writer(file)
        while True:
            ret, frame = cap.read()
            if not ret:
                    break    
            if frame_num % fps == 0:
                t1 = time.time()
                frame = cv2.resize(frame,  (720, 1080))
                frame = unsharpmask(frame, 0)
                frame = cv2.convertScaleAbs(frame * 255)
                #frame = cv2.rotate(frame, cv2.ROTATE_180)
                cv2.imshow('frame', frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                result = ocr.ocr(frame)
                result = result[0]
                print(result)
                try: 
                    boxes = [line[0] for line in result]
                    scores = [line[-1][1] for line in result]
                    txts = [(line[-1][0]) for line in result]   
                    line = [frame_num]
                    for num in range(len(txts)):
                        print(txts[num])
                        print(str(scores[num]))
                        line.append("\'" + txts[num])
                        line.append(scores[num])
                    writer.writerow(line)
                except:
                    writer.writerow([frame_num])
                print('Time: ', time.time() - t1)
            frame_num += 1
        cap.release()
    
def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--video_paths", default='./videos', help="Path to folder_videos",
    )
    random.seed(0)
    args = parser.parse_args()
    video_paths = os.listdir(args.video_paths)
    video_picks = []
    for i in range(5):
        video_pick = random.choice(video_paths)
        video_picks.append(video_pick)
        process_video(os.path.join('./videos', video_pick))
    print(video_picks)

if __name__ == "__main__":
    main()
