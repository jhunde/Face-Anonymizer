# Face Anonymizer

## Installation / Import Requirements 
Add the following texts into `requirement.txt`
```txt
opencv-python==4.10.0.84
mediapipe==0.10.18
```
Then run `pip install -r requirement.txt`


> **Note:** In the `main.py` file make sure to include these modules/libraries:
```py
import cv2 as cv
import os
import mediapipe as mp
import argparse
```

## Image Processing

```py
def process_img(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            # This is needed but why?
            x1, y1, w, h = int(x1 * W), int(y1 * H), int(w * W), int(h * H)

            # Blur faces
            img[y1 : y1 + h, x1 : x1 + w, :] = cv.blur(
                img[y1 : y1 + h, x1 : x1 + w, :], (40, 40)
            )

    return img
```

## Face Detection
[Mediapipe Face Detection](https://mediapipe.readthedocs.io/en/latest/solutions/face_detection.html)
> **Note:** Mediapipe `FaceDetection` parameters
> + `model_selector` is an integer ranging from `0` to `1`. By default it's set to `0` if not specified.
>   + `0` - short-range model that works best for faces within 2 meters from the camera
>   + `1` - full-range model best for faces within 5 meters  
>
> + `min_detection_confidence` value (`[0.0, 1.0]`) from the face detection model for the detection to be considered successful. Default to `0.5`

```py
# Detect faces
mp_face_detection = mp.solutions.face_detection
```

### Image Face Detection
**Before** image blur | **After** image blur
:---: | :---: |
<img src='./data/face.jpg' alt="Before image blur" width= "210" height="290"/> |  <img src="./output/blur_img.jpg" alt="After image blur" width= "210" height="290"/>|


### Vidoe Face Detection



### Webcam Face Detection

## To Do List
+ [X] Finish creating a bounding box 
+ [x] Save the blurred image file
+ [x] Finish blurring the `face.mp4`
+ [x] Troubleshoot the video blurring
+ [x] Blur webcam
+ [ ] Complete README file 

### Research
+ [ ] How does img array works?
+ [ ] Why is line 27 in `main.py` needed? And how does it work?

## Resources/References
Photo by <a href="https://unsplash.com/@princearkman?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Prince Akachi</a> on <a href="https://unsplash.com/photos/smiling-man-wearing-black-turtleneck-shirt-holding-camrea-4Yv84VgQkRM?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>
      
Video by [Mikhail Nilov from Pexels:](https://www.pexels.com/video/a-woman-talking-at-the-podium-8731389/)
