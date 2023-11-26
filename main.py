import ultralytics
from ultralytics import YOLO
import cv2


#loading a model:

model = YOLO("D:/Fayed/UTM/Sem8/FYP2/dataset/generated_model/best.pt")
results = model.predict(source=0, show=True, conf=0.8)

#cv2.waitKey(5000)  # Giving time for visualizing the predicted objects in the image
#cv2.destroyAllWindows()
print(results[0])
print("//////////////////////////////////////")

print(results[0].boxes)
print("//////////////////////////////////////")

print(results[0].boxes.cls)
print("//////////////////////////////////////")
bottles = 0
for i in results[0].boxes.cls:
    if i==0:
        bottles+=1

bottles_missing = 24 - bottles


print("//////////////////////////////////////")

img = results[0].orig_img


# # Mouse callback function
# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click event
#         print(f"Left button clicked at ({x}, {y})")
#
# # Example image (replace this with your image data)
# #image_path = 'path/to/your/image.jpg'
# #img = cv2.imread(image_path)  # Load the image using OpenCV
#
# # Create a window and display the image
# cv2.imshow('Image with Mouse Callback', img)
#
# # Set the mouse callback function for the window
# cv2.setMouseCallback('Image with Mouse Callback', mouse_callback)

# Wait for any key press and then close the window

#(29, 41)
# Function to display visual alert
def show_alert(message):
    cv2.putText(img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if bottles_missing:
    show_alert(f"{bottles_missing} bottles are missing")
