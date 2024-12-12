import cv2 as cv
import os
import mediapipe as mp
import argparse


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

            # Visualize bbox_img
            # img = cv.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 5)

            # Blur faces
            img[y1 : y1 + h, x1 : x1 + w, :] = cv.blur(
                img[y1 : y1 + h, x1 : x1 + w, :], (40, 40)
            )

    return img


def deallocate_memory(videoCampture):
    videoCampture.release()
    cv.destroyAllWindows()
    return


args = argparse.ArgumentParser()
args.add_argument("--mode", default="webcam")  # {"image", "video", "webcam"}
args.add_argument("--filePath", default="./data")

args = args.parse_args()


# Create output folder if it doesn't exist
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
) as face_detection:

    if args.mode in ["image"]:
        print(f"Image mode: {args.mode}")
        # Read image
        img = cv.imread(os.path.join(args.filePath, "face.jpg"))
        blur_img = process_img(img, face_detection)

        cv.imshow("video blurred", blur_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # Save image
        cv.imwrite(os.path.join(output_dir, "blur_img.jpg"), blur_img)

    if args.mode in ["video"]:
        print(f"Video mode: {args.mode}")
        cap = cv.VideoCapture(os.path.join(args.filePath, "face.mp4"))
        ret, frame = cap.read()

        # Inputs for cv2.VideoWriter
        output_file_location = os.path.join(output_dir, "output.mp4")
        codec = cv.VideoWriter_fourcc(*"MP4V")
        fps = 25
        frameSize = (frame.shape[1], frame.shape[0])

        output_video = cv.VideoWriter(output_file_location, codec, fps, frameSize)

        while ret:
            frame = process_img(frame, face_detection)
            output_video.write(frame)
            ret, frame = cap.read()

        deallocate_memory(cap)

    if args.mode in ["webcam"]:
        webcam = cv.VideoCapture(1, cv.CAP_DSHOW)
        ret, frame = webcam.read()

        while ret:
            frame = process_img(frame, face_detection)
            cv.imshow("webcam", frame)

            # 33ms per frame
            if cv.waitKey(33) & 0xFF == ord("q"):
                break

            ret, frame = webcam.read()

        deallocate_memory(webcam)
