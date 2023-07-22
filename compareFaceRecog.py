from datetime import timedelta
from firebase_admin import initialize_app,credentials,storage, db

from saimeseNetwork import siamese_model
from supportFunc import url_to_image,preprocess_image,saimese_pairs  

cred = credentials.Certificate("./zainab-alert025-key.json")
firebase_app = initialize_app(
    cred, {'storageBucket': 'zainab-alert025.appspot.com',
           'databaseURL': 'https://zainab-alert025-default-rtdb.firebaseio.com/'
           })
main_model = siamese_model()

def compare_found_missing_faces_optimized(found_id):
    
    bucket = storage.bucket(app=firebase_app)

    results = []
    dic = {"ChildFound": [], "ChildMissing":[]}
    
    data_child_found = db.reference('ChildFoundInfo/'+found_id).get()
    data_childern_missing=db.reference('ChildMissingInfo/').get()

    for imgPath in data_child_found['imagePath']: 
        image_url_1 = bucket.blob(imgPath).generate_signed_url(
                    timedelta(seconds=10000), method='GET')
        
        image_array=url_to_image(image_url_1)
        if image_array is None:
            continue
        preprocess_image_array = preprocess_image(image_array)
        if preprocess_image_array is None:
            continue

        for child_missing_id, data_child_missing in data_childern_missing.items():
            print("data_child_missing",data_child_missing)
            for path in data_child_missing['imagePath']:
                image_url_2 = bucket.blob(path).generate_signed_url(
                            timedelta(seconds=10000), method='GET')

                pairs = saimese_pairs(image_url_2)
                pairs[0][0, :, :, :] = preprocess_image_array

                dist = main_model.predict([pairs[0], pairs[1]])[0][0]
                results.append((child_missing_id,image_url_2, dist))

        results.sort(key=lambda x: x[2])   

        data_child_found["images_path"] = [image_url_1]    
        dic["ChildFound"].append({found_id: data_child_found})    

        missing_id = results[0][0]
        data_child_missing = data_childern_missing[missing_id]
        data_child_missing["images_path"] = [image_url_2]   
        dic["ChildMissing"].append({missing_id: data_child_missing})
    print("END",dic )
    return dic