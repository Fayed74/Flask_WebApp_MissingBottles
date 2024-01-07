import math

from ultralytics import YOLO
import cv2

def video_detection(path_x):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)

    # cap.get(N) retrieve a ceratin property of VideoCapture
    # cap.set(N, value) setting a ceratin property of VideoCapture

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # FPS = int(cap.get(5))
    # current_frame_number = int(cap.get(1))

    # model = YOLO("D:/Fayed/UTM/Sem8/FYP2/dataset/generated_model/best.pt")
    #///
    # model = YOLO("../yolov8n.pt")
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
    # while True:
    #     ret, frame = cap.read()
    #     print(f"vid frame: {frame}")
    #
    #     if not ret:
    #         break
    #
    #     results = model(frame, stream=True)
    #     for r in results:
    #         boxes = r.boxes
    #         for box in boxes:
    #             x1, y1, x2, y2 = box.xyxy[0]
    #             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #             print(x1, y1, x2, y2)
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
    #             conf = math.ceil((box.conf[0] * 100)) / 100
    #             cls = int(box.cls[0])
    #             class_name = classNames[cls]
    #             label = f'{class_name}{conf}'
    #             t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    #             # print(f">>>>{t_size}")
    #             c2 = x1 + t_size[0], y1 - t_size[1] - 3
    #             cv2.rectangle(frame, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
    #             cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    #///

    model = YOLO("D:/Fayed/UTM/Sem8/FYP2/dataset/generated_model/best.pt")

    frame_counter = 0
    skip_frames = 5  # Process every 5th frame

    while True:
        ret, frame = cap.read()
        print(f"vid frame: {frame}")

        if not ret:
            break

        frame_counter += 1

        if frame_counter % skip_frames != 0:
            continue  # Skip frames if not the desired frame to process

        results = model.predict(
            source=frame,
            show=True, conf=0.8)

        bottles = 0
        for detection in results:
            boxes_cls_list = list(detection.boxes.cls)
            for i in boxes_cls_list:
                if i == 0:
                    bottles += 1

        bottles_missing = 24 - bottles

        img = frame  # Assuming 'frame' is the original frame

        def show_alert(message):
            cv2.putText(img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if bottles_missing > 1:
            show_alert(f"{bottles_missing} bottles are missing")
        elif bottles_missing == 1:
            show_alert(f"{bottles_missing} bottle is missing")
        elif bottles_missing == 0:
            cv2.putText(img, "Package ready", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif bottles_missing < 0:
            cv2.putText(img, f"Watch out! more than required by {bottles_missing*-1} ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        cv2.imshow("Image", frame)
        yield img
        yield frame

    cv2.destroyAllWindows()

