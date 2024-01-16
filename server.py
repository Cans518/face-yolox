#!/usr/bin/env python
from flask import Flask, request, jsonify
import base64
from Person import Person
import cv2
import numpy as np

# Create the Flask app
app = Flask(__name__)

# Initialize the Person object
p = Person()

# Name: detect
# Description: Detects the people and gestures in the image
@app.route('/api/v1/detect', methods=['POST'])
def detect():
    # Get the image from the request
    image_b64 = (str)(request.json['image'])
    # Decode the image
    # image = base64.b64decode(image_b64[21:]) # [21:] to remove the "data:image/jpeg;base64," part
    image = base64.b64decode(image_b64)
    # Convert to OpenCV format
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    # Detect
    image_out, name_date = p.detect_image(image)
    # Return the strucutre
    return jsonify({
        'image': image_out,
        'name_date': name_date
    })

if __name__ == '__main__':
    # Run the app
    app.run(host='demo.mzee.top', port=8080, debug=False)
