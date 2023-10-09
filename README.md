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
cd into each PMS or Machinery folder and run
```
python3 main.py --input <path2input image> --maps <path2maps detect table>
```

## Output
Dict contains OCR results according to the index typed in the target file: `./data/target.png`

