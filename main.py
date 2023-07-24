from flask import Flask, request
from flask_ngrok import run_with_ngrok
from compareFaceRecog import compare_found_missing_faces_optimized,process_video_and_upload_faces,compare_found_missing_faces_all_optimized
import time

app = Flask(__name__)
run_with_ngrok(app)

@app.route('/compare_face', methods=['GET'])
def view_compare_found_missing_faces_optimized():
    # Get form data from URL
    child_found_id = request.args.get('url_1', '')
    print(child_found_id)
    
    # Process the form data
    start= time.time()
    result = compare_found_missing_faces_optimized(child_found_id)
    end= time.time()
    print("End-View: TIME: ", end - start)

    return result

@app.route('/compare_face_all', methods=['GET'])
def view_compare_found_missing_faces_all_optimized():
    # Process the form data
    start= time.time()
    result = compare_found_missing_faces_all_optimized()
    end= time.time()
    print("End-View: TIME: ", end - start)

    return result


@app.route('/result_video', methods=['GET'])
def view_process_video_upload_faces():

    # http://127.0.0.1:8000/result/?url_1=ali-found-ID&remote_img_count=5
    child_id = request.args.get('url_1', '')
    start= time.time()
    result_video = process_video_and_upload_faces(child_id, remote_img_count = 5)
    end= time.time()
    print("End-View-Video: TIME: ", end - start)
    return result_video

if __name__ == '__main__':
    app.run()