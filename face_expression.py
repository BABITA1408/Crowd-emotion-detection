import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

# Load the image
# image = cv2.imread('images/get_together.jpg')
# image = cv2.imread('images/large_grp.jpg')
image = cv2.imread('images/project_pic.jpg')
# image = cv2.imread('images/large_grp2.jpg')
# image = cv2.imread('images/large_grp3.jpeg')
# image = cv2.imread('images/large_grp4.jpeg')
# image = cv2.imread('images/sm1missing.jpeg')
# image = cv2.imread('images/smthng_wrong.jpg')
# image = cv2.imread('images/smthng_wrong2.jpg')
# image = cv2.imread('images/solo.jpeg')
# image = cv2.imread('images/two_people.jpeg')

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
print('Number of detected faces:', len(faces))
num_faces = len(faces)


# Crop faces and store in an array
cropped_faces = []
for (x, y, w, h) in faces:
    cropped_faces.append(image[y:y+h, x:x+w])

# Display the cropped faces
# for face in cropped_faces:
#     cv2.imshow('Cropped Face', face)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

angry = 0
disgust = 0
fear = 0
happy = 0
sad = 0
surprise = 0
neutral = 0

for element in cropped_faces:
    result = DeepFace.analyze(
        element, actions=['emotion'], enforce_detection=False)
    emotion = result[0]['dominant_emotion']
    print(emotion)

    if emotion == 'angry':
        angry = angry+1
    elif emotion == 'disgust':
        disgust = disgust+1
    elif emotion == 'fear':
        fear = fear+1
    elif emotion == 'happy':
        happy = happy+1
    elif emotion == 'sad':
        sad = sad+1
    elif emotion == 'surprise':
        surprise = surprise+1
    else:
        neutral = neutral+1

emotion = {}
emotion['angry'] = angry
emotion['disgust'] = disgust
emotion['fear'] = fear
emotion['happy'] = happy
emotion['sad'] = sad
emotion['surprise'] = surprise
emotion['neutral'] = neutral

max_key = max(emotion, key=lambda k: emotion[k])

# Print the key with the maximum value
print("dominant emotion is",max_key)


if num_faces == 1:
    print("Message: This is a solo picture.")
elif num_faces == 2:
    if happy == 2:
        print("Message: This is a picture of a couple.")
    else:
        print("Message: This is a picture of two people.")
elif num_faces <= 4:
    if happy == num_faces:
        print("Message: This is a picture of friends or family.")
    elif happy == num_faces - 1 and neutral == 1:
        print("Message: This is a picture of friends or family with someone who doesn't want to be there.")
    else:
        print("Message: This is a picture of a small group of people.")
elif num_faces <= 6:
    if happy == num_faces:
        print("Message: Get together of friends")
else:
    if happy == num_faces:
        print("Message: This is a picture of a party or celebration.")
    elif happy >= 2 and happy + neutral == num_faces:
        print("Message: This is a picture of a party or celebration with someone who doesn't want to be there.")
    elif happy >= 2 and happy + neutral == num_faces - 1:
        print("Message: This is a picture of a party or celebration with someone missing.")
    elif sad + neutral + fear + disgust + angry == num_faces:
        print("Message: something wrong !! ")
    else:
        print("Message: This is a picture of a large group of people.")
