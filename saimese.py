import cv2
import numpy as np
import tensorflow as tf
from urllib import request
from tensorflow.keras import backend as K
import datetime
from datetime import timedelta
from saimese_nw import siamese_model_func
from firebase_admin import initialize_app,credentials,storage, db
cred = credentials.Certificate("./zainab-alert025-key.json")

firebase_app = initialize_app(
    cred, {'storageBucket': 'zainab-alert025.appspot.com',
           'databaseURL': 'https://zainab-alert025-default-rtdb.firebaseio.com/'
           })

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 0.5
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

# # Register the custom loss function
tf.keras.utils.get_custom_objects()['contrastive_loss'] = contrastive_loss
my_model = tf.keras.models.load_model("./siam-face-recognition-046.h5",custom_objects={'contrastive_loss': contrastive_loss})

# my_model=siamese_model_func()

def url_to_image(url):

    resp = request.urlopen(url)  # download the image from the URL
    image = np.asarray(bytearray(resp.read()), dtype="uint8")  # convert the image to a NumPy array
    try:
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)  # read the image into OpenCV format
        return image
    except Exception as e:
        return None  # return None if an error occurs during image decoding

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image):   
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_face = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5, minSize=(5, 5))
    print(gray_face)
    if gray_face == ():
      return None
    (x, y, w, h) = gray_face[0]
    face_img = gray_img[y:y+h, x:x+w]

    img = cv2.resize(face_img, (160, 160))
    img = np.array(img, dtype=np.float64)
    img /= 255
    img = img[..., np.newaxis]
    return img

def saimese_pairs(url_1,url_2):
    image=url_to_image(url_1)
    image_test = url_to_image(url_2)
    
    if (image is None) or (image_test is None):
        return (False, 1)  
    
    pairs = [np.zeros((1, 160, 160, 1)) for _ in range(2)]
    pairs[0][0, :, :, :] = preprocess_image(image)
    pairs[1][0, :, :, :] = preprocess_image(image_test)
    return pairs

def compare_found_missing_faces_optimized(found_id):
    
    bucket = storage.bucket(app=firebase_app)

    results = []
    dic = {"ChildFound": {}, "ChildMissing": {}}
    data_child_found = db.reference('ChildFoundInfo/'+found_id).get()
    data_childern_missing=db.reference('ChildMissingInfo/').get()

    image_url_1 = bucket.blob(data_child_found['imagePath'][0]).generate_signed_url(
                timedelta(seconds=10000), method='GET')
    print(image_url_1)

    for child_missing_id, data_child_missing in data_childern_missing.items():
        print("data_child_missing",data_child_missing)
        for path in data_child_missing['imagePath']:
            image_url_2 = bucket.blob(path).generate_signed_url(
                        timedelta(seconds=10000), method='GET')

            pairs = saimese_pairs(image_url_1, image_url_2)
            dist =my_model.predict([pairs[0], pairs[1]])[0][0]
            results.append((child_missing_id,image_url_2, dist))

    results.sort(key=lambda x: x[2])   
    print("END",results[0])
    return results[0]