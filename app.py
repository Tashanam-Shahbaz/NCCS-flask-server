from flask import Flask, request
from saimese import compare_found_missing_faces_optimized

app = Flask(__name__)

print("hello")
@app.route('/process', methods=['GET'])
def process_form():
    # Get form data from URL
    child_found_id = request.args.get('url_1', '')
    print(child_found_id)
    
    # Process the form data
    result = compare_found_missing_faces_optimized(child_found_id)

    return result

if __name__ == '__main__':
    app.run(debug=True)