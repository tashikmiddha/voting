# import cv2
# import pickle
# import numpy as np
# import os 

# if not os.path.exists('data/'):
#     os.makedirs('data/')

# video=cv2.VideoCapture(0)
# facedetect=cv2.CascadeClassifier(cv2.data.haarcascades+'harcascade.xml')
# faces_data=[]
# i=0
# name=input("enter your adhar number: ")
# framestotal=51
# captureafterframe=2
# while True:
#     ret,frame=video.read()
#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     faces=facedetect.detectMultiScale(gray,1.3,5)
#     for (x,y,w,h) in faces:
#         crop_img=frame[y:y+h,x:x+w]
#         resized_img=cv2.resize(crop_img,(50,50))
#         if len(faces_data)<=framestotal and i%captureafterframe==0:
#             faces_data.append(resized_img)
#         i=i+1
#         cv2.putText(frame,str(len(faces_data)),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),1)
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),1)

#     cv2.imshow('frame',frame)
#     k=cv2.waitKey(1)
#     if k==ord('q') or len(faces_data)>=framestotal:
#         break
# video.release()
# cv2.destroyAllWindows()

# faces_data=np.asarray(faces_data)
# faces_data=faces_data.reshape((framestotal,-1))


# if 'names.pkl' not in os.listdir('data/'):
#     names=[name]*framestotal
#     with open('data/names.pkl','wb')as f:
#         pickle.dump(names,f)
# else:
#     with open('data/names.pkl','rb')as f:
#         names=pickle.load(f)
#     names=names+[name]*framestotal
#     with open('data/names.pkl','wb')as f:
#         pickle.dump(names,f)


# if 'faces_data.pkl' not in os.listdir('data/'):
#     with open ('data/faces_data.pkl','wb') as f:
#         pickle.dump(faces_data,f)
# else:
#      with open('data/faces_data.pkl','rb')as f:
#         faces=pickle.load(f)
#      faces=np.append(faces,faces_data,axis=0)
#      with open('data/faces_data.pkl','wb')as f:
#         pickle.dump(faces,f)


import cv2
import pickle
import numpy as np
import os

# Ensure the 'data/' directory exists
if not os.path.exists('data/'):
    os.makedirs('data/')

# Initialize video capture and face detector
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces_data = []
frame_count = 0
name = input("Enter your Aadhar number: ")
framestotal = 51
captureafterframe = 2

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture video. Exiting...")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) < framestotal and frame_count % captureafterframe == 0:
            faces_data.append(resized_img)
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
    
    frame_count += 1
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q') or len(faces_data) >= framestotal:
        break

video.release()
cv2.destroyAllWindows()

# Convert face data to a numpy array and reshape
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape((framestotal, -1))

# Save or update names data
names_file = 'data/names.pkl'
if not os.path.exists(names_file):
    names = [name] * framestotal
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
else:
    with open(names_file, 'rb') as f:
        names = pickle.load(f)
    names.extend([name] * framestotal)
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)

# Save or update faces data
faces_file = 'data/faces_data.pkl'
if not os.path.exists(faces_file):
    with open(faces_file, 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open(faces_file, 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open(faces_file, 'wb') as f:
        pickle.dump(faces, f)

print("Data saved successfully.")
