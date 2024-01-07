import math

from ultralytics import YOLO
import cv2

def image_detection(path_x):
    frame = path_x
    print(f"img frmae1: {frame}")
    frame = cv2.imread(frame)
    print(f"img frmae2: {frame}")

    model = YOLO("D:/Fayed/UTM/Sem8/FYP2/dataset/generated_model/best.pt")

    results = model.predict(
        source=frame,
        show=True, conf=0.8)

    bottles = 0
    for i in results[0].boxes.cls:
        if i == 0:
            bottles += 1

    bottles_missing = 24 - bottles

    print("//////////////////////////////////////")

    img = results[0].orig_img

    def show_alert(message):
        cv2.putText(img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    if bottles_missing > 1:
        show_alert(f"{bottles_missing} bottles are missing")
    elif bottles_missing == 1:
        show_alert(f"{bottles_missing} bottle is missing")
    elif bottles_missing == 0:
        cv2.putText(img, "Package ready", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 10), 3)
    elif bottles_missing < 0:
        cv2.putText(img, f"Watch out! more than required by {bottles_missing * -1} ", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)


    cv2.imshow("Image", frame)
    yield img
# cv2.destroyAllWindows()
