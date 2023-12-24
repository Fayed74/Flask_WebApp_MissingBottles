import math

from ultralytics import YOLO
import cv2

bottles_missing = 0
def image_detection(path_x):
    img = path_x

    #cap = cv2.VideoCapture(video_capture)

    # cap.get(N) retrieve a ceratin property of VideoCapture
    # cap.set(N, value) setting a ceratin property of VideoCapture

    #frame_width = int(cap.get(3))
    #frame_height = int(cap.get(4))
    # FPS = int(cap.get(5))
    # current_frame_number = int(cap.get(1))

    # model = YOLO("D:/Fayed/UTM/Sem8/FYP2/dataset/generated_model/best.pt")
    model = YOLO("D:/Fayed/UTM/Sem8/FYP2/dataset/generated_model/best.pt")

    results = model.predict(
        source=img,
        show=True, conf=0.8)

    bottles = 0
    for i in results[0].boxes.cls:
        if i == 0:
            bottles += 1

    bottles_missing = 24 - bottles

# Function to display visual alert
def show_alert(message):
    cv2.putText(img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    yield img
    cv2.destroyAllWindows()



if bottles_missing:
    show_alert(f"{bottles_missing} bottles are missing")
