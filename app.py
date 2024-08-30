from flask import Flask, request, jsonify
import face_recognition
import numpy as np
from PIL import Image
import json
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# Global variable to store face encodings
encodings = {}

def update_encodings(data):
    global encodings
    encodings = {}
    for item in data:
        try:
            # Convert string encoding to list of floats
            face_encoding = np.array(json.loads(item['image_person']), dtype=np.float64)
            encodings[item['id']] = face_encoding
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Error decoding encoding data: {str(e)}")
            continue
    return encodings

@app.route('/')
def index():
    return "Hello World"

@app.route('/api/v1/face_register', methods=['POST'])
def register_face():
    allData = request.form.to_dict()

    update_encodings(json.loads(allData['all_data']))

    if 'image_person' not in request.files:
        return jsonify({
            "message": "No image_person file in request"
        }), 404

    image_person = request.files['image_person']

    try:
        image = Image.open(image_person).convert('RGB')
        image_np = np.array(image)

        face_encodings = face_recognition.face_encodings(image_np)
        if not face_encodings:
            return jsonify({
                "message": "No face detected"
            }), 400

        face_encoding = face_encodings[0].tolist()

        return jsonify({
            "status": "success",
            "message": "Face registered successfully",
            "image_person": face_encoding
        })
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({
            "message": f"Error processing image: {str(e)}"
        }), 500
        
@app.route('/api/v1/face_recognition', methods=['POST'])
def face_recognition_match():
    global encodings
    
    allData = request.form.to_dict()

    data = json.loads(allData['all_data'])
    update_encodings(data)
    
    logging.info(f'encodings: {encodings}')
    
    if 'image_person' not in request.files:
        return jsonify({
            "message": "No image_person file in request"
        }), 404

    image_person = request.files['image_person']

    try:
        image = Image.open(image_person).convert('RGB')
        image_np = np.array(image)

        face_encodings = face_recognition.face_encodings(image_np)
        if not face_encodings:
            return jsonify({
                "message": "No face detected"
            }), 404

        face_encoding = face_encodings[0]
        known_encodings = list(encodings.values())
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        
        if True in matches:
            matched_index = matches.index(True)
            matched_id = list(encodings.keys())[matched_index]
            return jsonify({
                "status": "success",
                "id_person": matched_id
            })
        else:
            return jsonify({
                "status": "no_match",
                "message": "No matching face found"
            }), 400

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({
            "message": f"Error processing image: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
