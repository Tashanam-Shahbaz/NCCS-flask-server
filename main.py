from flask import Flask, request
from flask_ngrok import run_with_ngrok
from compareFaceRecog import compare_found_missing_faces_optimized


app = Flask(__name__)
run_with_ngrok(app)

@app.route('/compareface', methods=['GET'])
def view_compare_found_missing_faces_optimized():
    # Get form data from URL
    child_found_id = request.args.get('url_1', '')
    print(child_found_id)
    
    # Process the form data
    result = compare_found_missing_faces_optimized(child_found_id)

    return result

if __name__ == '__main__':
    app.run()