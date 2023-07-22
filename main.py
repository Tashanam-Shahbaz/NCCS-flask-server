from flask import Flask, request
from flask_ngrok import run_with_ngrok
from compareFaceRecog import compare_found_missing_faces_optimized,process_video_and_upload_faces
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

@app.route('/result_video', methods=['GET'])
def view_process_video_upload_and_compare_faces(request):

    # http://127.0.0.1:8000/result/?url_1=ali-found-ID&remote_img_count=5
    child_found_id = request.args.get('url_1', '')
    result_video = process_video_and_upload_faces(child_found_id, remote_img_count = 20)
    if result_video["Success"]:
        result=compare_found_missing_faces_optimized(child_found_id)   
    return result

if __name__ == '__main__':
    app.run()