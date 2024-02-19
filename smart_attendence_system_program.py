import cv2
import numpy as np
import face_recognition as face_rec
import os
import pyttsx3 as textSpeach
from datetime import datetime

engine = textSpeach.init()

def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

def dynamic_tolerance(num_faces):
    if num_faces > 1:
        return 0.6  # Adjust the threshold for face clustering if multiple faces are detected
    else:
        return 0.5  # Default threshold for a single face

# Use the current working directory as the base path
base_path = os.getcwd()


def findEncoding(images):
    known_face_encodings = []
    for img in images:
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_rec.face_locations(img)
        encodings = face_rec.face_encodings(img, face_locations)
        known_face_encodings.extend(encodings)
    return known_face_encodings

 #It records names and timestamps in a CSV file and generates a welcome message.
def MarkAttendance(name):
    filename = 'attendance.csv'
    header = 'Name, Time\n'

    # Check if the file exists
    if not os.path.isfile(filename):
        with open(filename, 'w') as f:
            f.write(header)

    with open(filename, 'a+') as f:
        myDatalist = f.readlines()
        nameList = [entry.split(',')[0] for entry in myDatalist]

        if name not in nameList:
            now = datetime.now()
            timestr = now.strftime('%H:%M')
            f.writelines(f'{name}, {timestr}\n')
            statement = f'Welcome to class, {name}'
            engine.say(statement)
            engine.runAndWait()

# : It captures video frames, detects faces, compares them with known faces, marks attendance, and displays the video stream with marked faces and names.
studentImg = []
studentName = []
myList = os.listdir(base_path)
for cl in myList:
    # Check if the file is an image (assuming it has an extension like .jpg, .jpeg, .png)
    if cl.lower().endswith(('.jpg', '.jpeg', '.png')):
        curimg = cv2.imread(os.path.join(base_path, cl))
        studentImg.append(curimg)
        studentName.append(os.path.splitext(cl)[0])

EncodeList = findEncoding(studentImg)

vid = cv2.VideoCapture(0)
vid.set(3, 1280)  # Set width
vid.set(4, 720)   # Set height
vid.set(5, 30)    # Set frame rate

while True:
    success, frame = vid.read()
    Smaller_frames = cv2.resize(frame, (0, 0), None, 0.25, 0.25)

    facesInFrame = face_rec.face_locations(Smaller_frames)
    encodeFacesInFrame = face_rec.face_encodings(Smaller_frames, facesInFrame)

    for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame):
        num_faces = len(facesInFrame)
        tolerance = dynamic_tolerance(num_faces)
        facedis = face_rec.face_distance(EncodeList, encodeFace)
        matchIndex = np.argmin(facedis)

        if facedis[matchIndex] < tolerance:
            matches = face_rec.compare_faces([EncodeList[matchIndex]], encodeFace, tolerance=tolerance)
            if matches[0]:
                name = studentName[matchIndex].upper()
                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                MarkAttendance(name)

    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

