import cv2 as cv
import os
import mediapipe as mp


# Read image
img = cv.imread(os.path.join("data", "face.jpg"))

# cv.imshow("img", img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
) as face_detection:
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detection is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

# Blur faces

# Save image
