import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

# Load the image
image = cv2.imread('images/img11.jpg')

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
print('Number of detected faces:', len(faces))

# Crop faces and store in an array
cropped_faces = []
for (x, y, w, h) in faces:
    cropped_faces.append(image[y:y+h, x:x+w])

# Display the cropped faces
for face in cropped_faces:
    cv2.imshow('Cropped Face', face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

angry=0
disgust=0
fear=0
happy=0
sad=0
surprise=0
neutral=0

for element in cropped_faces:
    result = DeepFace.analyze(element, actions = ['emotion'], enforce_detection=False)
    emotion = result[0]['dominant_emotion']
   
    if emotion=='angry':
        angry=angry+1
    elif emotion=='disgust':
        disgust=disgust+1
    elif emotion=='fear':
        fear=fear+1
    elif emotion=='happy':
        happy=happy+1
    elif emotion=='sad':
        sad=sad+1
    elif emotion=='surprise':
        surprise=surprise+1
    else:
        neutral=neutral+1

avinya = {}
avinya['angry']=angry
avinya['disgust']=disgust
avinya['fear']=fear
avinya['happy']=happy
avinya['sad']=sad
avinya['surprise']=surprise
avinya['neutral']=neutral

max_key = max(avinya, key=lambda k: avinya[k])

# Print the key with the maximum value
print(max_key)

