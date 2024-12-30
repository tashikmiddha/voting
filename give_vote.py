# from sklearn.neighbors import KNeighborsClassifier
# import cv2
# import pickle 
# import numpy as np
# import os
# import csv
# import time
# from datetime import datetime
# from win32com.client import Dispatch

# def speck(str):
#     speak=Dispatch(("SAPI.SpVoice"))
#     speak.Speak(str)

# video=cv2.VideoCapture(0)
# facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# with open('data/names.pkl','rb') as f:
#     labels=pickle.load(f)
# with open('data/faces_data.pkl','rb') as f:
#     faces=pickle.load(f)

# knn=KNeighborsClassifier(n_neighbors=5)

# knn.fit(faces,labels)

# imgbackground=cv2.imread("background.png")

# col_name=['NAME','VOTE','DATE','TIME']

# while True:
#     ret,frame=video.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)
#     for (x, y, w, h) in faces:
#        crop_img = frame[y:y + h, x:x + w]
#        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1,-1)
#        output=knn.predict(resized_img)
#        ts=time.time()
#        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M:%S")
#        exist=os.path.isfile("votes"+".csv")
#        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
#        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
#        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), -1)
#        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (225, 225, 255), 1)
#        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
#        attendance=[output[0],timestamp]
#     imgbackground[370:370+400,225:225+640]=frame  
#     cv2.imshow('frame',imgbackground)
#     k=cv2.waitkey(1)

#     def check_if_existes(value):
#         try:
#             with open("votes.csv",'r') as csvfile:
#                 reader=csv.reader(csvfile)
#                 for row in reader:
#                     if row and row[0]==value:
#                         return True
#         except FileNotFoundError:
#             print("Unable to open the csv file")
#         return False
#     voter_exist=check_if_existes(output[0])
#     if voter_exist:
#         speak("you have aleady voted")
#         break

#     if k==ord('1'):
#        speck("your vote has been recorded")
#        time.sleep(3)
#        if exist:
#            with open("votes"+".csv","+a")as csvfile:
#                writer=csv.writer(csvfile)
#                attendance=[output[0],"Group 1",date,timestamp]
#                writer.writerow(attendance)
#            csvfile.close()
#        else :
#            with open("votes"+".csv","+a")as csvfile:
#                writer=csv.writer(csvfile)
#                writer.writerow(col_name)
#                attendance=[output[0],"Group 1",date,timestamp]
#                writer.writerow(attendance)
#            csvfile.close()
#         speak("thank you for partisipating in the election")
#        break
    # if k==ord('2'):
    #    speck("your vote has been recorded")
    #    time.sleep(3)
    #    if exist:
    #        with open("votes"+".csv","+a")as csvfile:
    #            writer=csv.writer(csvfile)
    #            attendance=[output[0],"Group 2",date,timestamp]
    #            writer.writerow(attendance)
    #        csvfile.close()
    #    else :
    #        with open("votes"+".csv","+a")as csvfile:
    #            writer=csv.writer(csvfile)
    #            writer.writerow(col_name)
    #            attendance=[output[0],"Group 2",date,timestamp]
    #            writer.writerow(attendance)
    #        csvfile.close()
    #     speak("thank you for partisipating in the election")
    #    break
    # if k==ord('3'):
    #    speck("your vote has been recorded")
    #    time.sleep(3)
    #    if exist:
    #        with open("votes"+".csv","+a")as csvfile:
    #            writer=csv.writer(csvfile)
    #            attendance=[output[0],"Group 3",date,timestamp]
    #            writer.writerow(attendance)
    #        csvfile.close()
    #    else :
    #        with open("votes"+".csv","+a")as csvfile:
    #            writer=csv.writer(csvfile)
    #            writer.writerow(col_name)
    #            attendance=[output[0],"Group 3",date,timestamp]
    #            writer.writerow(attendance)
    #        csvfile.close()
    #     speak("thank you for partisipating in the election")
    #    break
    # if k==ord('4'):
    #    speck("your vote has been recorded")
    #    time.sleep(3)
    #    if exist:
    #        with open("votes"+".csv","+a")as csvfile:
    #            writer=csv.writer(csvfile)
    #            attendance=[output[0],"Group 4",date,timestamp]
    #            writer.writerow(attendance)
    #        csvfile.close()
    #    else :
    #        with open("votes"+".csv","+a")as csvfile:
    #            writer=csv.writer(csvfile)
    #            writer.writerow(col_name)
    #            attendance=[output[0],"Group 4",date,timestamp]
    #            writer.writerow(attendance)
    #        csvfile.close()
    #     speak("thank you for partisipating in the election")
    # break

# video.release()
# cv2.destroyAllWindows()

from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import subprocess

# Function to speak text on macOS
def speck(text):
    subprocess.run(["say", text])  # Uses macOS's built-in "say" command

# Load video and face detection model
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained KNN model and labels
with open('data/names.pkl', 'rb') as f:
    labels = pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    faces = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces, labels)

imgbackground = cv2.imread("background.png")


# Resize the video feed to fit within the blank space in the background
video_feed_width, video_feed_height = 750, 500  # Adjust these dimensions as per the blank space in your image
video_feed_x, video_feed_y = 30, 200  # Coordinates of the top-left corner of the blank space
col_name = ['NAME', 'VOTE', 'DATE', 'TIME']

# Function to check if the voter already exists in the CSV
def check_if_exists(value):
    try:
        with open("votes.csv", 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == value:
                    return True
    except FileNotFoundError:
        print("Unable to open the CSV file")
    return False

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        voter_exist = check_if_exists(output[0])
        if voter_exist:
            speck("You have already voted")
            time.sleep(3)
            video.release()
            cv2.destroyAllWindows()
            exit()
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist = os.path.isfile("votes.csv")

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (225, 225, 255), 1)
        attendance = [output[0], timestamp]

        

        if cv2.waitKey(1) == ord('1'):
            speck("Your vote has been recorded")
            time.sleep(3)
            if exist:
                with open("votes.csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    attendance = [output[0], "Group 1", date, timestamp]
                    writer.writerow(attendance)
            else:
                with open("votes.csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(col_name)
                    attendance = [output[0], "Group 1", date, timestamp]
                    writer.writerow(attendance)
            speck("Thank you for participating in the election")
            time.sleep(3)
            video.release()
            cv2.destroyAllWindows()
            exit()
        if cv2.waitKey(1) == ord('2'):
            speck("Your vote has been recorded")
            time.sleep(3)
            if exist:
                with open("votes.csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    attendance = [output[0], "Group 2", date, timestamp]
                    writer.writerow(attendance)
            else:
                with open("votes.csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(col_name)
                    attendance = [output[0], "Group 2", date, timestamp]
                    writer.writerow(attendance)
            speck("Thank you for participating in the election")
            time.sleep(3)
            video.release()
            cv2.destroyAllWindows()
            exit()
        if cv2.waitKey(1) == ord('3'):
            speck("Your vote has been recorded")
            time.sleep(3)
            if exist:
                with open("votes.csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    attendance = [output[0], "Group 3", date, timestamp]
                    writer.writerow(attendance)
            else:
                with open("votes.csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(col_name)
                    attendance = [output[0], "Group 3", date, timestamp]
                    writer.writerow(attendance)
            speck("Thank you for participating in the election")
            time.sleep(3)
            video.release()
            cv2.destroyAllWindows()
            exit()
        if cv2.waitKey(1) == ord('4'):
            speck("Your vote has been recorded")
            time.sleep(3)
            if exist:
                with open("votes.csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    attendance = [output[0], "Group 4", date, timestamp]
                    writer.writerow(attendance)
            else:
                with open("votes.csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(col_name)
                    attendance = [output[0], "Group 4", date, timestamp]
                    writer.writerow(attendance)
            speck("Thank you for participating in the election")
            time.sleep(3)
            video.release()
            cv2.destroyAllWindows()
            exit()
        

    # imgbackground[370:370 + 400, 225:225 + 640] = frame
    # Resize frame to match the target region in imgbackground
    # resized_frame = cv2.resize(frame, (640, 398))
    # imgbackground[370:370 + 398, 225:225 + 640] = resized_frame
    resized_frame = cv2.resize(frame, (video_feed_width, video_feed_height))
    imgbackground[video_feed_y:video_feed_y + video_feed_height, video_feed_x:video_feed_x + video_feed_width] = resized_frame


    cv2.imshow('frame', imgbackground)

    if cv2.waitKey(1) == 27:  # Escape key to exit
        break


video.release()
cv2.destroyAllWindows()



