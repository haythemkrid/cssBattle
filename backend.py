from flask import Flask, request, jsonify
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import io

app = Flask(__name__)

def read_image(file):
    # Read file into a numpy array
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def calculate_ssim(img1, img2):
    # Convert to grayscale for SSIM
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

@app.route('/compare', methods=['POST'])
def compare_images():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Please upload image1 and image2'}), 400

    img1 = read_image(request.files['image1'])
    img2 = read_image(request.files['image2'])

    if img1.shape != img2.shape:
        return jsonify({'error': 'Images must have the same size'}), 400

    similarity_score = calculate_ssim(img1, img2)
    return jsonify({'similarity': similarity_score})

if __name__ == '__main__':
    app.run(debug=True)
