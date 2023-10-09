# bemac_ocr
OCR LCD device

## Input
The input consists of 3 arguments

|arg|help|
|`--input`|Path to input video file|
|---------|------------------------|
|`--output`|Path to output video file|
|`--maps`|Path to map_detect file|

## Installation
```
pip3 install -r requirements.txt
```
## Run
cd into each PMS or Machinery folder and run
```
python3 main.py --input <path2input video> --output <path2output video> --maps <path2maps detect table>
```

## Output
Output video: `./data/output_video.avi`

