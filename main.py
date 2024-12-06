import cv2 as cv
import os
import mediapipe as mp

# Create output folder if it doesn't exist
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read image
img = cv.imread(os.path.join("data", "face.jpg"))
H, W, _ = img.shape

# Detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
) as face_detection:
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            # This is needed but why?
            x1, y1, w, h = int(x1 * W), int(y1 * H), int(w * W), int(h * H)

            # Visualize bbox_img
            # img = cv.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 5)

            # Blur faces
            img[y1 : y1 + h, x1 : x1 + w, :] = cv.blur(
                img[y1 : y1 + h, x1 : x1 + w, :], (40, 40)
            )

blur_img = img
cv.imshow("img", img)
cv.waitKey(0)
cv.destroyAllWindows()
print(blur_img.shape)
# Save image
cv.imwrite(os.path.join(output_dir, "blur_img.jpg"), blur_img)
