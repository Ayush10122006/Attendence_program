import cv2  
import face_recognition 
import numpy as np  
import csv 
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

bill_image = face_recognition.load_image_file("/home/ayush/Attendence_program/face attendance system/code/Bill.png")
bill_encoding = face_recognition.face_encodings(bill_image)[0]

elon_image = face_recognition.load_image_file("/home/ayush/Attendence_program/face attendance system/code/elon.jpg")
elon_encoding = face_recognition.face_encodings(elon_image)[0]

photo_image = face_recognition.load_image_file("/home/ayush/Attendence_program/face attendance system/code/photo.jpg")
photo_encoding = face_recognition.face_encodings(photo_image)[0]




known_face_encoding = [
    bill_encoding,
    photo_encoding,  
    elon_encoding,
]

known_face_names = [
    'Bill Gates',
    'Ayush',
    'Elon Musk',
    
]

students = known_face_names.copy()


now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
filename = current_date + '.csv'


file_exists = os.path.isfile(filename)

f = open(filename, 'a', newline='')  
inwriter = csv.writer(f)


if not file_exists:
    inwriter.writerow(["Name", "Date", "Time", "Status"])

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("ERROR")
        break

 
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        name = "Unknown"  

        face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        
        if name != "Unknown" and name in students:
            students.remove(name)
            current_time = datetime.now().strftime("%H:%M:%S")
            inwriter.writerow([name, current_date, current_time, "Present"])
            print(f" Attendance marked for {name} at {current_time}")
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)

    
    cv2.imshow("Attendance System", frame)

    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
f.close()
print("🅿️")