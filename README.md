# Recognize license plate in Vietnam
This is a project for recognizing Vietnamese license plates (both single-line and double-line).
First, you can clone this repository and run the following command to install the required libraries
## Set up environments
```
pip install -r requirements.txt
```

## How to use 
You can run it immediately using the command below, where link_to_image is the URL of the image containing the license plate.
```
python main.py -i link_to_image 
```

## Result
   This license plate recognition project can work with both single-line and double-line license plates. It can even read license plates that are partially obscured in some cases.
   ![alt text](<result/Screenshot 2025-01-07 at 09.23.12.png>)
However, it still has some drawbacks :worried:

## Drawbacks
* If the license plate is too small compared to the image, it may fail to recognize the characters.
* Sometimes, it mistakenly identifies letters, such as confusing B with D.
*  Performs poorly when the image is too blurry.

## Credit 
* The model for detecting the location of the license plate in the image uses YOLOv8. The configuration file can be found at [roboflow-projects](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4)
* The character recognition process is referenced and developed from [Nhận diện biển số xe Việt Nam](https://viblo.asia/p/nhan-dien-bien-so-xe-viet-nam-Do754P9L5M6)