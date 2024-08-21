from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from deepface import DeepFace

app = Flask(__name__)

def process_image(image):
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    
    # Detect faces
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    print('Number of detected faces:', len(faces))
    num_faces = len(faces)

    cropped_faces = []
    for (x, y, w, h) in faces:
        cropped_faces.append(image[y:y+h, x:x+w])

    # Analyze emotions
    emotion_counts = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}

    for face in cropped_faces:
        result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
        emotion_counts[dominant_emotion] += 1

    # Determine the dominant emotion
    max_emotion = max(emotion_counts, key=emotion_counts.get)

    # Message generation based on the number of faces and emotions
    if num_faces == 1:
        message = "This is a solo picture."
    elif num_faces == 2:
        if emotion_counts['happy'] == 2:
            message = "This is a picture of a couple."
        else:
            message = "This is a picture of two people."
    elif num_faces <= 4:
        if emotion_counts['happy'] == num_faces:
            message = "This is a picture of friends or family."
        elif emotion_counts['happy'] == num_faces - 1 and emotion_counts['neutral'] == 1:
            message = "This is a picture of friends or family with someone who doesn't want to be there."
        else:
            message = "This is a picture of a small group of people."
    elif num_faces <= 6:
        if emotion_counts['happy'] == num_faces:
            message = "Get together of friends."
        else:
            message = "This is a picture of a group."
    else:
        if emotion_counts['happy'] == num_faces:
            message = "This is a picture of a party or celebration."
        elif emotion_counts['happy'] >= 2 and emotion_counts['happy'] + emotion_counts['neutral'] == num_faces:
            message = "This is a picture of a party or celebration with someone who doesn't want to be there."
        elif emotion_counts['sad'] + emotion_counts['neutral'] + emotion_counts['fear'] + emotion_counts['disgust'] + emotion_counts['angry'] == num_faces:
            message = "Something wrong !!"
        else:
            message = "This is a picture of a large group of people."

    return message, max_emotion

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        npimg = np.fromfile(file, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        message, dominant_emotion = process_image(image)

        return render_template('result.html', message=message, dominant_emotion=dominant_emotion)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
