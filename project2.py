

import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

#Load known faces

#Kushal's Data
#harrys_image = face_recognition.load_image_file("faces/harry.jpg")
kushal_image = face_recognition.load_image_file("faces/kushal.jpeg")

#Kushal_encoding = face_recognition.face_encodings(harrys_image)[0]
kushal_encoding = face_recognition.face_encodings(kushal_image)[0]

#Ayush's Data
ayush_image = face_recognition.load_image_file("faces/ayush2.jpeg")
ayush_encoding = face_recognition.face_encodings(ayush_image)[0]

#Shantanu's Data
shantanu_image = face_recognition.load_image_file("faces/shantanu.jpeg")
shantanu_encoding = face_recognition.face_encodings(shantanu_image)[0]

#Mayan's Data
mayan_image = face_recognition.load_image_file("faces/mayan.jpeg")
mayan_encoding = face_recognition.face_encodings(mayan_image)[0]

#Satvir Sir's Data
satvirsir_image = face_recognition.load_image_file("faces/satvirsir.jpeg")
satvirsir_encoding = face_recognition.face_encodings(satvirsir_image)[0]

#Amit Gupta sir data
AmitGuptasir_image = face_recognition.load_image_file("faces/Amit Gupta sir.jpg")
AmitGuptasir_encoding = face_recognition.face_encodings(AmitGuptasir_image)[0]

known_face_encodings = [kushal_encoding, ayush_encoding , shantanu_encoding, mayan_encoding, satvirsir_encoding, AmitGuptasir_encoding ]
known_face_names = ["Kushal ", "Ayush ", "Shantanu ", "Mayan ", "Satvir Sir ","Amit Gupta Sir "]

#list of expected students
students = known_face_names.copy()

face_locations =[]
face_encodings = []

#Get the current date and time

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f= open(f"{current_date}.csv", "w+",newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    #Recognize facess
    face_locations =face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if (matches[best_match_index]):
            name = known_face_names[best_match_index]

            # Add the text if a person is present
            if name in known_face_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2
                cv2.putText(frame, name + "Present", bottomLeftCornerOfText, font,
                            fontScale, fontColor, thickness, lineType)

                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()



