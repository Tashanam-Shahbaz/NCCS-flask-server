import os 
import shutil

from datetime import timedelta
from firebase_admin import initialize_app,credentials,storage, db

from saimeseNetwork import siamese_model
from supportFunc import url_to_image,preprocess_image,saimese_pairs,face_capture_from_video

cred = credentials.Certificate("./firebaseConnection/zainab-alert025-key.json")
firebase_app = initialize_app(
    cred, {'storageBucket': 'zainab-alert025.appspot.com',
           'databaseURL': 'https://zainab-alert025-default-rtdb.firebaseio.com/'
           })
main_model = siamese_model()

def compare_found_missing_faces_optimized(found_id):
    
    bucket = storage.bucket(app=firebase_app)

    
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
        
        results = []
        for child_missing_id, data_child_missing in data_childern_missing.items():
            print("data_child_missing",data_child_missing)
            for path in data_child_missing['imagePath']:
                image_url_2 = bucket.blob(path).generate_signed_url(
                            timedelta(seconds=10000), method='GET')

                pairs = saimese_pairs(image_url_2)
                pairs[0][0, :, :, :] = preprocess_image_array

                dist = main_model.predict([pairs[0], pairs[1]])[0][0]
                results.append((child_missing_id,image_url_2, dist))
        
        results=results.sort(key=lambda x: x[2])   
        print(results)

        data_child_found["images_path"] = [image_url_1]    
        dic["ChildFound"].append({found_id: data_child_found}) 

        data_child_found["images_path"] = [image_url_1]    
        dic["ChildFound"].append({found_id: data_child_found})   

        data_child_found["images_path"] = [image_url_1]    
        dic["ChildFound"].append({found_id: data_child_found})      


        missing_id_1 = results[0][0]
        data_child_missing = data_childern_missing[missing_id_1]
        data_child_missing["images_path"] = [results[0][1]] 
        dic["ChildMissing"].append({missing_id_1: data_child_missing})

        missing_id_2 = results[1][0]
        data_child_missing = data_childern_missing[missing_id_2]
        data_child_missing["images_path"] = [results[0][1]] 
        dic["ChildMissing"].append({missing_id_2: data_child_missing})

        missing_id_3 = results[2][0]
        data_child_missing = data_childern_missing[missing_id_3]
        data_child_missing["images_path"] = [results[0][1]] 
        dic["ChildMissing"].append({missing_id_3: data_child_missing})
        
    print("END",dic )
    return dic


def process_video_and_upload_faces(found_id, remote_img_count):

    bucket = storage.bucket(app=firebase_app)
    found_child_video_lst = list(bucket.list_blobs(
        prefix=f"ChildFound/Video/{found_id}"))

    temp_image_dir = "temp_Image"
    temp_video_dir = "temp_Video"

    # Create temporary folders if they don't exist
    os.makedirs(temp_video_dir, exist_ok=True)
    os.makedirs(temp_image_dir, exist_ok=True)

    try:
        for video_count, video_name in enumerate(found_child_video_lst):
            local_video_path = f"{temp_video_dir}/{found_id}_{video_count}.mp4"
            remote_video_path = str(video_name).split(",")[1][1:]

            blob = bucket.blob(remote_video_path)
            blob.download_to_filename(local_video_path)
            blob.delete()

            face_capture_from_video(local_video_path, temp_image_dir, video_count, remote_img_count)

        lst_images=os.listdir(temp_image_dir)
        
        db_ref = db.reference('/ChildFoundInfo/'+found_id)
        remote_image_lst=["ChildFound/Image/" + found_id + "/" + filename for filename in lst_images]
        db_ref.child("imagePath").set(remote_image_lst)
        
        for i in lst_images:
            local_file_path = f"{temp_image_dir}/{i}"
            remote_file_path = f"ChildFound/Image/{found_id}/{i}"

            image_blob = bucket.blob(remote_file_path)
            image_blob.upload_from_filename(local_file_path)

    except Exception as e:
        print(f'Error occurred: {str(e)}')
        # Clean up temporary folders
        shutil.rmtree(temp_image_dir, ignore_errors=True)
        shutil.rmtree(temp_video_dir, ignore_errors=True)
        return {"Success": False}

    finally:
        # Clean up temporary folders
        shutil.rmtree(temp_image_dir, ignore_errors=True)
        shutil.rmtree(temp_video_dir, ignore_errors=True)

    return {"Success": True}