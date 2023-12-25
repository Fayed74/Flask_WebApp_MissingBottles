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

    # bottles = 0
    # for i in results[0].boxes.cls:
    #     if i == 0:
    #         bottles += 1
    # bottles_missing = 24 - bottles
    #
    # if bottles_missing:
    #     message = f"{bottles_missing} bottles are missing"
    #     cv2.putText(img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #
    # yield img

    ##

    # for frame in results.render():  # Iterate over rendered frames
    #     bottles = 0
    #     for i in frame.boxes.cls:
    #         if i == 0:
    #             bottles += 1
    #     bottles_missing = 24 - bottles
    #
    #     if bottles_missing:
    #         message = f"{bottles_missing} bottles are missing"
    #         cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #
    #     yield frame

    # classNames = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
    #               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    #               'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    #               'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    #               'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    #               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    #               'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
    #               'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    #               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    #               'teddy bear', 'hair drier', 'toothbrush']
    #
    # for r in results:
    #     boxes = r.boxes
    #     for box in boxes:
    #         x1, y1, x2, y2 = box.xyxy[0]
    #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #         print(x1, y1, x2, y2)
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
    #         conf = math.ceil((box.conf[0] * 100)) / 100
    #         cls = int(box.cls[0])
    #         class_name = classNames[cls]
    #         label = f'{class_name}{conf}'
    #         t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    #         # print(f">>>>{t_size}")
    #         c2 = x1 + t_size[0], y1 - t_size[1] - 3
    #         cv2.rectangle(frame, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
    #         cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    ##
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

    if bottles_missing:
        show_alert(f"{bottles_missing} bottles are missing")

    # cv2.imshow("Image", frame)
    yield img
# cv2.destroyAllWindows()
