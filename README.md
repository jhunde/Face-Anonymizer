# Face Anonymizer

## Installation Requirements 
Add the following texts into `requirement.txt`
```txt
opencv-python==4.10.0.84
mediapipe==0.10.18
```
Then run `pip install -r requirement.txt`

## Face Detection
[Mediapipe Face Detection](https://mediapipe.readthedocs.io/en/latest/solutions/face_detection.html)
> **Note:** Mediapipe `FaceDetection` parameters
> + `model_selector` is an integer ranging from `0` to `1`. By default it's set to `0` if not specified.
>   + `0` - short-range model that works best for faces within 2 meters from the camera
>   + `1` - full-range model best for faces within 5 meters  
>
> + `min_detection_confidence` value (`[0.0, 1.0]`) from the face detection model for the detection to be considered successful. Default to `0.5`

**Before** image blur | **After** image blur
:---: | :---: |
<img src='./data/face.jpg' alt="Before image blur" width= "210" height="290"/> |  <img src="./output/blur_img.jpg" alt="After image blur" width= "210" height="290"/>|



## To Do List
+ [X] Finish creating a bounding box 
+ [x] Save the blurred image file
+ [x] Finish blurring the `face.mp4`
+ [x] Troubleshoot the video blurring
+ [x] Blur webcam

### Research
+ [ ] How does img array works?
+ [ ] Why is line 27 in `main.py` needed? And how does it work?

## Resources/References
Photo by <a href="https://unsplash.com/@princearkman?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Prince Akachi</a> on <a href="https://unsplash.com/photos/smiling-man-wearing-black-turtleneck-shirt-holding-camrea-4Yv84VgQkRM?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>
      
Video by [Mikhail Nilov from Pexels:](https://www.pexels.com/video/a-woman-talking-at-the-podium-8731389/)
