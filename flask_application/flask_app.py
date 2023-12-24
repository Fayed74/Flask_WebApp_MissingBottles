from flask import Flask, Response, jsonify, request, render_template, session
import cv2
from yolo_video import video_detection
from yolo_image import image_detection

# FlaskForm is required to receive input from the user, like uploading a video file to our object detection model
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, StringField, DecimalRangeField, IntegerRangeField
from werkzeug.utils import secure_filename
# validators to make sure the user enter the right input format
from wtforms.validators import InputRequired, NumberRange
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Fayed'
app.config['UPLOAD_FOLDER'] = 'static/files'

# using FlaskForm to get input video file from user
class UploadFileForm(FlaskForm):
    """We store the uploaded video file path in the FileField in the variable file
        we have added validators to make sure the user inputs the video in the valid format,
        and the user does upload the video when prompted to do so"""
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")


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

def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear() # to remove the video uploaded before
    return render_template('indexproject.html')
"""Rendering the webacam rage
    Now lets make a webcam page for the application
    Use 'app.route()' method to render the webcam page at '/webcam' """

@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    session.clear()
    return render_template('ui.html')
@app.route("/FrontPage", methods=['GET', 'POST'])
def front():
    # Upload File Form: Create an instance for the Upload File Form
    form = UploadFileForm()
    if form.validate_on_submit():
        # our uploaded video file path is saved here
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename))) # Then save the file
        # Use session storage to save video file path and confidence value
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('videoprojectnew.html', form=form)

@app.route('/webapp')
def webapp():
    return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video')
def video():
    return Response(generate_frames(path_x = session.get('video_path', None)), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)