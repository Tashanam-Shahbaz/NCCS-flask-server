from turtle import distance
import cv2
import numpy as np
from urllib import request
from importlib.resources import path
# from deepface import DeepFace
import face_recognition
import os
from typing import Tuple,Optional,Union

def url_to_image(url: str) -> Optional[np.ndarray]:

    resp = request.urlopen(url)  # download the image from the URL
    image = np.asarray(bytearray(resp.read()), dtype="uint8")  # convert the image to a NumPy array
    try:
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)  # read the image into OpenCV format
        return image
    except Exception as e:
        return None  # return None if an error occurs during image decoding
    
def my_face_recognition(image, image_test):
    # Can use for local images
    # image = face_recognition.load_image_file(url_1)
    # image_test = face_recognition.load_image_file(url_2)
    
    # image = url_to_image(url_1)
    # image_test = url_to_image(url_2)
    # if (image is None) or (image_test is None):
    #     return (False, 1)  
    try:  # return if it doesnot found any faces in image.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # faceloc = face_recognition.face_locations(image)[0]
        encode = face_recognition.face_encodings(image)[0]
    except Exception as e:
        print(e)
        return (False, 1)
    # cv2.rectangle(image, (faceloc[3], faceloc[0]),
    #               (faceloc[1], faceloc[2]), (255, 0, 255), 2)
    try:
        image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
        # faceLocTest = face_recognition.face_locations(image_test)[0]
        encodeTest = face_recognition.face_encodings(image_test)[0]
    except Exception as e:
        print(e)
        return (False, 1)
    
    # cv2.rectangle(image_test, (faceLocTest[3], faceLocTest[0]),
    #               (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

    results = face_recognition.compare_faces([encode], encodeTest, 0.6)
    faceDis = face_recognition.face_distance([encode], encodeTest)
    # print(results, faceDis)
    # cv2.putText(image_test, f'{results} {round(faceDis[0], 2)}', (
    #     50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # cv2.imshow("window_1",image)
    # cv2.imshow("window_2",image_test)
    # cv2.waitKey(0)

    return (results[0], faceDis[0])

# a="https://storage.googleapis.com/zainab-alert025.appspot.com/ChildMissing/Ahmed-Lost-2292/1.jpg?Expires=1673310336&GoogleAccessId=firebase-adminsdk-ihxrn%40zainab-alert025.iam.gserviceaccount.com&Signature=XvOQgtxUUzyqsZ9B0q8wpUaaFaR4MA0bQoCaNH3HLu%2F0I%2Bs7O2ea0JTzqubrzfz3e9Imk4nxu5YivnO833UMfeY45jM5cOgGcFY9e%2BrJnnPryp85jfFzllbVG56QxhM7nwDnqNDMvErgzDUd2%2FQTGJXRm9WgMM%2FYS7dQteTrkFUGS5%2ByAVK42v3s8z886ex88lQ4tHbj9QPST%2FqR4hVwk1QkZLFzXH9Og7KrmRKGdfeBeF%2B8ouldQWrOVrixyrscjfVK4RfCimwiNZQibOQqRNa1SHz4WiErPdNf62lE9p%2BqDEfbXTy6A6jQ8jUFliWZChxBJTt2ijpDEVhJnyskog%3D%3D"
# b="https://storage.googleapis.com/zainab-alert025.appspot.com/ChildMissing/Ahmed-Lost-2292/2.jpg?Expires=1673310339&GoogleAccessId=firebase-adminsdk-ihxrn%40zainab-alert025.iam.gserviceaccount.com&Signature=huXy1nu0A2pNPfHrm3ELMvbw7JRMVFZE%2B03o6%2BJ%2Fj6k6MjIFmmSc8f18rw77EvWldnQLVmsLZJLSSh5nRHAeiPyN8%2FULc3JMK7Gf0nWkG0Jn7DZzIlGmfPFJdVanWh9A3DWVccMZoPswGqSmDcqpoXW3ryo4GmGfbYF5WerIcC7Mh3scZWr9u%2FIBowKekQrgnTDjzsLCNgqGn5LWzZnSGvnioIcFlq0kNwlcUhYGdVPZEbMnUZ81y1OIdsH8T4ANJK6SxwJDv0f5DoO%2Fde81jmVppciqtjKphfAkn9cB9zhoLlhAwclGeB4vLq%2F026Ht6hhloyZ5bUZNz%2F%2F6OwmtpQ%3D%3D"

# t=myfacerecognition(a,b)
# print(t)

 # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
 # Convert to grayscale
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# faces = face_cascade.detectMultiScale(gray, 1.05, 5)


def face_capture_from_video(video_path, output_dir, video_count, image_count=5):
    # Load the video file
    video = cv2.VideoCapture(video_path)
    # Set a counter to keep track of the number of frames processed
    count = 0

    # Loop through the video frames
    while video.isOpened():
        # Read the next frame from the video
        ret, frame = video.read()
        if ret:
            rgb_frame = frame[:, :, ::-1]
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame)
            # Save the detected faces
            for location in face_locations:
                top, right, bottom, left = location
                face_image = frame[top:bottom, left:right]
                print(face_image)
                cv2.imwrite(
                    f'{output_dir}/face_{video_count}_{count}.jpg', face_image)
                count += 1
        else:
            pass
        if count >= image_count:
            break
    # Release the video and close all windows
    video.release()
    cv2.destroyAllWindows()
    return