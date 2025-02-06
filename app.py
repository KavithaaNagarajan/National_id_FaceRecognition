from flask import Flask, request, jsonify, render_template
import cv2
import torch
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
from pdf2image import convert_from_path
import tempfile

# Specify the path to Poppler
poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"

app = Flask(__name__)

# Load YOLO model (pretrained on general objects including humans)
model = YOLO("yolov8n.pt")  # Use yolov8s.pt or better for accuracy

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML frontend

@app.route('/upload', methods=['POST'])
def upload():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "No images uploaded!"}), 400

    # Get the uploaded images
    image1 = request.files['image1']
    image2 = request.files['image2']

    # Save the uploaded images temporarily
    image1_path = os.path.join('uploads', 'image1.png')
    image2_path = os.path.join('uploads', 'image2.png')
    image1.save(image1_path)
    image2.save(image2_path)

    # Check if image1 is a PDF, if yes process as PDF
    if image1.filename.lower().endswith(".pdf"):
        # Convert PDF to image and process it
        result = compare_pdf_with_image(image1_path, image2_path)
    else:
        # Process image as usual
        result = compare_images(image1_path, image2_path)

    # Return the similarity score as a response
    return jsonify(result)

def compare_images(image1_path, image2_path):
    # Read the first image (National ID)
    image1 = cv2.imread(image1_path)

    # Detect objects in the image
    results = model(image1)

    face_crop = None
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            cls = int(box.cls[0].item())  # Class index

            if cls == 0:  # Class 0 is 'person' (which may include the face)
                face_crop = image1[y1:y2, x1:x2]  # Crop the detected area
                break
        if face_crop is not None:
            break

    if face_crop is None:
        return {"error": "No face detected in the first image."}

    # Read the second image (for comparison)
    image2 = cv2.imread(image2_path)

    # Ensure both images are of the same size for SSIM comparison
    face_crop_resized = cv2.resize(face_crop, (image2.shape[1], image2.shape[0]))

    # Convert images to grayscale for SSIM calculation
    gray_face_crop = cv2.cvtColor(face_crop_resized, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    similarity_index, _ = ssim(gray_face_crop, gray_image2, full=True)

    return {"similarity_score": similarity_index}

def compare_pdf_with_image(pdf_path, image2_path):
    # Create a temporary folder to store the PDF pages as images
    with tempfile.TemporaryDirectory() as temp_folder:
        # Convert the PDF to images using Poppler
        pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
        
        # For simplicity, assume we compare only the first page of the PDF
        first_page_image_path = os.path.join(temp_folder, 'page_1.jpg')
        pages[0].save(first_page_image_path, 'JPEG')

        # Now, compare the first page of the PDF (converted to an image) with the second uploaded image
        return compare_images(first_page_image_path, image2_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8893, threaded=True, debug=False)
