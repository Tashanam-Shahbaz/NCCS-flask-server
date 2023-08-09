import os 
import shutil

from datetime import timedelta
from firebase_admin import initialize_app,credentials,storage, db

from saimeseNetwork import siamese_model
from supportFunc import url_to_image,preprocess_image,saimese_pairs,face_capture_from_video
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    if data_child_found is not None and data_childern_missing is not None and 'imagePath' in data_child_found:
        for imgPath in data_child_found['imagePath']: 
            image_url_1 = bucket.blob(imgPath).generate_signed_url(
                        timedelta(seconds=10000), method='GET')

            image_array=url_to_image(image_url_1)
            if image_array is None:
                print("image_array",imgPath,image_url_1)
                continue
            preprocess_image_array = preprocess_image(image_array)
            if preprocess_image_array is None:
                print("preprocess_image_array",preprocess_image_array)
                continue
            results = []
            for child_missing_id, data_child_missing in data_childern_missing.items():
                print("data_child_missing",data_child_missing)
                if 'imagePath' in data_child_missing:
                    for path in data_child_missing['imagePath']:
                        image_url_2 = bucket.blob(path).generate_signed_url(
                                    timedelta(seconds=10000), method='GET')

                        pairs = saimese_pairs(image_url_2)
                        if pairs is None:
                            print("pairs",pairs)  
                            continue
                        
                        pairs[0][0, :, :, :] = preprocess_image_array

                        dist = main_model.predict([pairs[0], pairs[1]])[0][0]
                        results.append((child_missing_id,image_url_2, dist))

            results.sort(key=lambda x: x[2])

            if results[2]<0.01:
                data_child_found["images_path"] = [image_url_1]
                dic["ChildFound"].append({found_id: data_child_found})
                print("No Match Found.\nResult",results)
                return dic
            
            print("RESULts ",results)
            for i in range(3):
                missing_id = results[i][0]
                data_child_missing = data_childern_missing[missing_id]
                data_child_missing["images_path"] = [results[i][1]]
                dic["ChildMissing"].append({missing_id: data_child_missing})

                data_child_found["images_path"] = [image_url_1]
                dic["ChildFound"].append({found_id: data_child_found})

    print("END", dic)
    return dic


def compare_found_missing_faces_2_optimized(missing_id):

    bucket = storage.bucket(app=firebase_app)

    dic = {"ChildFound": [], "ChildMissing":[]}

    data_child_missing = db.reference('ChildMissingInfo/'+missing_id).get()
    data_childern_found=db.reference('ChildFoundInfo/').get()

    if data_child_missing is not None and data_childern_found is not None and 'imagePath' in data_child_missing:
        for imgPath in data_child_missing['imagePath']: 
            image_url_1 = bucket.blob(imgPath).generate_signed_url(
                        timedelta(seconds=10000), method='GET')

            image_array=url_to_image(image_url_1)
            if image_array is None:
                print("image_array",imgPath,image_url_1)
                continue
            preprocess_image_array = preprocess_image(image_array)
            if preprocess_image_array is None:
                print("preprocess_image_array",preprocess_image_array)
                continue
            results = []
            for child_found_id, data_child_found in data_childern_found.items():
                print("data_child_found",data_child_found)
                if 'imagePath' in data_child_found:
                    for path in data_child_found['imagePath']:
                        image_url_2 = bucket.blob(path).generate_signed_url(
                                    timedelta(seconds=10000), method='GET')

                        pairs = saimese_pairs(image_url_2)
                        if pairs is None:
                            print("pairs",pairs)  
                            continue
                        
                        pairs[0][0, :, :, :] = preprocess_image_array

                        dist = main_model.predict([pairs[0], pairs[1]])[0][0]
                        results.append((child_found_id,image_url_2, dist))

            results.sort(key=lambda x: x[2])   
            print("RESULts ",results)
            for i in range(3):
                found_id = results[i][0]
                data_child_found = data_childern_found[found_id]
                data_child_found["images_path"] = [results[i][1]]
                dic["ChildFound"].append({found_id: data_child_found})

                data_child_missing["images_path"] = [image_url_1]
                dic["ChildFound"].append({missing_id: data_child_missing})

    print("END", dic)
    return dic

def compare_found_missing_faces_all_optimized():
    
    bucket = storage.bucket(app=firebase_app)

    dic = {"ChildFound": [], "ChildMissing": []}
    
    data_childern_found = db.reference('ChildFoundInfo/').get()
    data_childern_missing = db.reference('ChildMissingInfo/').get()

    for child_found_id, data_child_found in data_childern_found.items():
        if 'imagePath' in data_child_found:
            for imgPath in data_child_found['imagePath']:
                image_url_1 = bucket.blob(imgPath).generate_signed_url(timedelta(seconds=10000), method='GET')
                
                image_array = url_to_image(image_url_1)
                if image_array is None:
                    continue
                preprocess_image_array = preprocess_image(image_array)
                if preprocess_image_array is None:
                    continue
                
                results = []
                for child_missing_id, data_child_missing in data_childern_missing.items():
                    print("data_child_missing",data_child_missing)
                    if 'imagePath' in data_child_missing:
                        for path in data_child_missing['imagePath']:
                            image_url_2 = bucket.blob(path).generate_signed_url(
                                        timedelta(seconds=10000), method='GET')

                            pairs = saimese_pairs(image_url_2)
                            if pairs is None:
                                continue
                            pairs[0][0, :, :, :] = preprocess_image_array

                            dist = main_model.predict([pairs[0], pairs[1]])[0][0]
                            results.append((child_missing_id,image_url_2, dist))
                    
                results.sort(key=lambda x: x[2])

                for i in range(3):
                    missing_id = results[i][0]
                    data_child_missing = data_childern_missing[missing_id]
                    data_child_missing["images_path"] = [results[i][1]]
                    dic["ChildMissing"].append({missing_id: data_child_missing})

                    data_child_found["images_path"] = [image_url_1]
                    dic["ChildFound"].append({child_found_id: data_child_found})

    print("END - compare_found_missing_faces_all_optimized ", dic)
    return dic


def process_video_and_upload_faces(child_id, remote_img_count):

    bucket = storage.bucket(app=firebase_app)
    id_type=child_id.split("-")[1]
    child_video_lst = list(bucket.list_blobs(
        prefix=f"Child{id_type}/Video/{child_id}"))

    temp_image_dir = "temp_Image"
    temp_video_dir = "temp_Video"

    # Create temporary folders if they don't exist
    os.makedirs(temp_video_dir, exist_ok=True)
    os.makedirs(temp_image_dir, exist_ok=True)

    try:
        for video_count, video_name in enumerate(child_video_lst):
            local_video_path = f"{temp_video_dir}/{child_id}_{video_count}.mp4"
            remote_video_path = str(video_name).split(",")[1][1:]

            blob = bucket.blob(remote_video_path)
            blob.download_to_filename(local_video_path)
            # blob.delete()

            face_capture_from_video(local_video_path, temp_image_dir, video_count, remote_img_count)

        lst_images=os.listdir(temp_image_dir)
        
        db_ref = db.reference(f"Child{id_type}Info/{child_id}")
        remote_image_lst=["Child"+id_type+"/Image/" + child_id + "/" + filename for filename in lst_images]
        db_ref.child("imagePath").set(remote_image_lst)
        
        for i in lst_images:
            local_file_path = f"{temp_image_dir}/{i}"
            remote_file_path = f"Child{id_type}/Image/{child_id}/{i}"

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


# PARALLEL PROCESSING

# def compare_images_parallel(preprocess_image_array, data_child_missing, bucket, results):
#     with ThreadPoolExecutor() as executor:
#         futures = [executor.submit(compare_image_distance, preprocess_image_array, path, bucket) 
#                    for path in data_child_missing['imagePath']]
#         for future in as_completed(futures):
#             result = future.result()
#             if result is not None:
#                 results.append(result)

# def compare_image_distance(preprocess_image_array, path, bucket):
#     image_url_2 = bucket.blob(path).generate_signed_url(timedelta(seconds=10000), method='GET')
#     pairs = saimese_pairs(image_url_2)
#     if pairs is not None:
#         pairs[0][0, :, :, :] = preprocess_image_array
#         dist = main_model.predict([pairs[0], pairs[1]])[0][0]
#         return (child_missing_id, image_url_2, dist)
#     return None