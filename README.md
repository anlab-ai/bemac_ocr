# bemac_ocr
OCR LCD device

## Input
The input consists of 3 arguments

|arg|help|
|---------|------------------------|
|`--input`|Path to input image file|
|`--maps`|Path to map_detect file|

## Installation
```
pip3 install -r requirements.txt
```
## Run
```
python3 bemac_detection.py --input <video folder input> --ouput <save ouput folder> --type <type devic, 0: machinery, 1:pms, 2:helicon, 3:CPPOder>
```

## Output
Dict contains OCR results according to the index typed in the target file: `./data/target.png`

