from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from categories import categories
from responses import responses
import random
import cv2
import numpy as np

app = Flask(__name__)


def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'  # Allow requests from any origin
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


app.after_request(add_cors_headers)

@app.route('/')

def hello_world():
    return 'Hello from Flask!'

@app.route('/grayscale', methods=['POST'])
def grayscaleimage():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        nparr = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid file'})

      
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred=cv2.GaussianBlur(gray,(5,5),0)
        edged_image= cv2.Canny(blurred, 30, 150)

        hist = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.equalizeHist(hist)

    
        retval, buffer = cv2.imencode('.jpg', processed_image)
        response_image = buffer.tobytes()

        print("success")

        return response_image, 200, {'Content-Type': 'image/jpeg'}

@app.route('/cartoon', methods=['POST'])
def cartoonImage():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        nparr = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid file'}), 400
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

   
        edges = cv2.Laplacian(blurred, cv2.CV_8U, ksize=5)

 
        _, mask = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY_INV)

        
        cartoon = cv2.bitwise_and(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), image)

        cartoon_image = cartoon

        retval, buffer = cv2.imencode('.jpg', cartoon_image)
        response_image = buffer.tobytes()


        return response_image, 200, {'Content-Type': 'image/jpeg'}
    
@app.route('/sketched', methods=['POST'])
def sketchedImage():
           
        if 'file' not in request.files:
            return 'No file part', 400

        file = request.files['file']
        nparr = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return 'Invalid file', 400

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   
        inverted_gray = 255 - gray

 
        blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)

  
        blended = cv2.divide(gray, 255 - blurred, scale=256)
        sketch = blended

      
        retval, buffer = cv2.imencode('.jpg', sketch)
        response_image = buffer.tobytes()

        return response_image, 200, {'Content-Type': 'image/jpeg'}

@app.route('/blur', methods=['POST'])
def blurredImage():
        if 'file' not in request.files:
            return 'No file part', 400

        file = request.files['file']
        nparr = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return 'Invalid file', 400

        blurred = cv2.medianBlur(image, 5) 
       
        blurred_image = blurred

       
        retval, buffer = cv2.imencode('.jpg', blurred_image)
        response_image = buffer.tobytes()

        return response_image, 200, {'Content-Type': 'image/jpeg'}
def determine_category(user_input):
    max_similarity = 0
    matched_category = None
    vectorizer = TfidfVectorizer()
    user_input_vector = vectorizer.fit_transform([user_input])
    for category, keywords in categories.items():
        category_keywords = ' '.join(keywords)
        category_vector = vectorizer.transform([category_keywords])
        similarity_score = cosine_similarity(user_input_vector, category_vector)[0][0]
        if similarity_score > max_similarity:
            max_similarity = similarity_score
            matched_category = category
    return matched_category

def get_responses(category):
    return random.choice(responses[category])

@app.route('/answer', methods=['POST'])
def chatbotresponse():
    try:
        data = request.get_json(force=True)
        user_prompt = data.get('user_prompt', '')
        category = determine_category(user_prompt)
        if category:
            responses = get_responses(category)
            response = {
                'status': 'success',
                'category': category,
              
                    'answer': responses,
                
                
            }
        else:
            response = {'status': 'failure', 'answer': "Sorry, I couldn't understand your request"}
        return jsonify(response)
    except Exception as e:
        return jsonify({'status': 'failure', 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
