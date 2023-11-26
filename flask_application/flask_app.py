from flask import Flask, Response, jsonify, request
import cv2
from yolo_video import video_detection

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Fayed'

#function generate_frames() takes path of input video file and gives us the output with bounding boxes

def generate_frames(path_x = ''):
    #yolo_output variable stores the output for each detection

    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        """Any flask application requires the encoded image to be converted into bytes,
        we will display the individual frames using yield keyword,
        we will loop over all individual frames and display them as video,
        when we want the individual frames to be replaced by the subsequent frames the Content-Type,
        or Mini-Type will be used
        """
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/webcam')
def webcam():
    return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video')
def video():
    return Response(generate_frames(path_x='../videos/bicycles.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)