from flask import Flask, request, jsonify, render_template
import os
from pdf2image import convert_from_path
import tempfile
from deepface import DeepFace
import tensorflow as tf
import traceback

# Set TensorFlow logging to avoid warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Specify the path to Poppler
poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"

app = Flask(__name__)

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

    # Validate image formats (allow PDFs as well)
    if not image1.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
        return jsonify({"error": "Invalid file type for image1. Only PNG, JPG, JPEG, PDF are allowed."}), 400
    if not image2.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
        return jsonify({"error": "Invalid file type for image2. Only PNG, JPG, JPEG, PDF are allowed."}), 400

    # Save the uploaded images temporarily
    image1_path = os.path.join('uploads', 'image1')
    image2_path = os.path.join('uploads', 'image2')

    # Ensure upload folder exists
    os.makedirs('uploads', exist_ok=True)

    image1.save(image1_path)
    image2.save(image2_path)

    # Check if both image1 and image2 are PDFs
    if image1.filename.lower().endswith(".pdf") and image2.filename.lower().endswith(".pdf"):
        # Convert both PDFs to images and compare them
        result = compare_pdfs(image1_path, image2_path)
    elif image1.filename.lower().endswith(".pdf"):
        # Convert only the first PDF to an image and compare with second image
        result = compare_pdf_with_image(image1_path, image2_path)
    elif image2.filename.lower().endswith(".pdf"):
        # Convert only the second PDF to an image and compare with first image
        result = compare_pdf_with_image(image2_path, image1_path)
    else:
        # Both files are images, process them directly
        result = compare_images(image1_path, image2_path)

    # Return the similarity score as a response
    return jsonify(result)


def compare_images(image1_path, image2_path):
    """
    Compares two images using DeepFace for face verification and returns the similarity score and verification result.
    """
    try:
        # Print paths to verify images
        print(f"Comparing images: {image1_path}, {image2_path}")

        # Verify if image files exist
        if not os.path.exists(image1_path):
            return jsonify({"error": f"Image1 does not exist at path: {image1_path}"}), 400
        if not os.path.exists(image2_path):
            return jsonify({"error": f"Image2 does not exist at path: {image2_path}"}), 400

        # Perform face verification using DeepFace with enforce_detection=False
        result = DeepFace.verify(image1_path, image2_path, enforce_detection=False)

        # Extract and print the similarity score (distance)
        similarity_score = result['distance']
        print(f"Similarity score (distance): {similarity_score}")

        # Check if the faces belong to the same person
        if result['verified']:
            message = "The faces belong to the same person."
        else:
            message = "The faces do not belong to the same person."
        
        print(message)

        # Return the similarity score, verification result, and message as JSON
        return {
            "similarity_score": similarity_score,
            "verified": result['verified'],
            "message": message  # Add the message to the response
        }
    
    except Exception as e:
        print(f"Error comparing images: {e}")
        # Log the error stack trace for better debugging
        traceback.print_exc()
        return {"error": f"An error occurred while processing the images: {e}"}


def compare_pdf_with_image(pdf_path, image_path):
    """
    Convert a PDF to image and compare the first page with the provided image using DeepFace.
    """
    try:
        # Create a temporary folder to store the PDF page as an image
        with tempfile.TemporaryDirectory() as temp_folder:
            # Convert the PDF to images using Poppler
            pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)

            # For simplicity, assume we compare only the first page of the PDF
            first_page_image_path = os.path.join(temp_folder, 'page_1.jpg')
            pages[0].save(first_page_image_path, 'JPEG')

            # Now, compare the first page of the PDF (converted to an image) with the provided image
            return compare_images(first_page_image_path, image_path)

    except Exception as e:
        print(f"Error comparing PDF with image: {e}")
        # Log the error stack trace for better debugging
        traceback.print_exc()
        return {"error": f"An error occurred while processing the PDF: {e}"}


def compare_pdfs(pdf1_path, pdf2_path):
    """
    Convert both PDFs to images and compare their first pages.
    """
    try:
        # Create a temporary folder to store the PDF pages as images
        with tempfile.TemporaryDirectory() as temp_folder:
            # Convert the first PDF to images
            pages1 = convert_from_path(pdf1_path, dpi=300, poppler_path=poppler_path)
            first_page_pdf1_image_path = os.path.join(temp_folder, 'pdf1_page_1.jpg')
            pages1[0].save(first_page_pdf1_image_path, 'JPEG')

            # Convert the second PDF to images
            pages2 = convert_from_path(pdf2_path, dpi=300, poppler_path=poppler_path)
            first_page_pdf2_image_path = os.path.join(temp_folder, 'pdf2_page_1.jpg')
            pages2[0].save(first_page_pdf2_image_path, 'JPEG')

            # Now, compare the first page of both PDFs (converted to images)
            return compare_images(first_page_pdf1_image_path, first_page_pdf2_image_path)

    except Exception as e:
        print(f"Error comparing PDFs: {e}")
        # Log the error stack trace for better debugging
        traceback.print_exc()
        return {"error": f"An error occurred while processing the PDFs: {e}"}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8894, threaded=True, debug=False)
