import cv2
import numpy as np
from urllib import request


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def url_to_image(url):  # download the image, convert it to a NumPy array, and then read it into OpenCV format

    resp = request.urlopen(url)  # download the image from the URL
    image = np.asarray(bytearray(resp.read()), dtype="uint8")  # convert the image to a NumPy array
    try:
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)  # read the image into OpenCV format
        return image
    except Exception as e:
        return None  # return None if an error occurs during image decoding


def preprocess_image(image): # preprocess the image for the model
    try:    
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_face = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5, minSize=(5, 5))
    except Exception as e:
      print(e)
      return None

    if gray_face == ():
        return None
    
    (x, y, w, h) = gray_face[0]
    face_img = gray_img[y:y+h, x:x+w]

    img = cv2.resize(face_img, (160, 160))
    img = np.array(img, dtype=np.float64)
    img /= 255
    img = img[..., np.newaxis]
    return img

def saimese_pairs(url_2):   # create pairs of images for the model
    
    # image=url_to_image(url_1)
    image_test = url_to_image(url_2)
    
    if image_test is None:
        return None
    
    img_w,img_h=160,160
    pairs = [np.zeros((1,img_w,img_h, 1)) for _ in range(2)]

    # pairs[0][0, :, :, :] = preprocess_image(image)
    temp =preprocess_image(image_test)
    if temp is None:
       return None  
    pairs[1][0, :, :, :] = temp
    return pairs


def face_capture_from_video(video_path, output_dir, video_count, image_count=5):    # capture faces from video and save them in a folder
    # Load the video file
    video = cv2.VideoCapture(video_path)
    # Set a counter to keep track of the number of frames processed
    count = 0

    # Loop through the video frames
    while video.isOpened():
        # Read the next frame from the video
        ret, frame = video.read()
        if ret:
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_face = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5, minSize=(5, 5))
            
            if gray_face == ():
                continue
            
            (x, y, w, h) = gray_face[0]
            face_img = frame[y:y+h, x:x+w]

            print(face_img)
            cv2.imwrite(
                f'{output_dir}/face_{video_count}_{count}.jpeg', face_img)
            count += 1
        else:
            pass
        if count >= image_count:
            break
    # Release the video and close all windows
    video.release()
    cv2.destroyAllWindows()
    return
