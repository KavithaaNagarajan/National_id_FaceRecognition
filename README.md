# National_id_FaceRecognition
![image](https://github.com/user-attachments/assets/e73939cf-a9e3-442a-ba7a-4fbd584a4e60)
![image](https://github.com/user-attachments/assets/56513774-1fea-409b-9cfd-9297d4df3e91)
![image](https://github.com/user-attachments/assets/d60e23fc-4e9b-4b36-ab74-dca18b63f539)
# Image and PDF Face Comparison API with YOLOv8 and SSIM
This is a Flask web application that allows users to upload two images (or one image and one PDF) to compare the similarity of faces detected in both. The application uses a YOLOv8 model for detecting faces and calculates the similarity between the detected faces using SSIM (Structural Similarity Index) for image comparison.

# Features
Face detection: Uses YOLOv8 to detect faces in images.
## Image similarity: Compares the faces using SSIM to determine how similar the images are.
## PDF support: If one of the uploaded files is a PDF, it is converted into an image (first page) for comparison.
## Flask API: Easy-to-use API for comparing images via a simple web interface.
# Technologies Used
## Flask: Web framework for building the API.
## YOLOv8 (Ultralytics): For object detection (specifically face detection).
## OpenCV: For image processing and manipulation.
## scikit-image: For SSIM (Structural Similarity Index) comparison.
## Poppler: For converting PDF files into images.
## Python 3.x: Programming language.
# Requirements
Before running the app, make sure you have the following installed:

Python 3.x
Flask
PyTorch (with CUDA support for GPU acceleration)
OpenCV
ultralytics
scikit-image
pdf2image
Poppler (for PDF conversion)


## Using the API
1. Upload Two Images for Comparison
Use the web interface to upload two images. The first image can either be a regular image or a PDF.
The second image should be a regular image.
The application will detect faces in the first image and compare the detected face with the second image. It will return a similarity score.

## API Endpoint:
Endpoint: /upload
Method: POST
## Parameters:
image1: The first image (can be an image or a PDF).
image2: The second image for comparison.
Example Request (via cURL):

# Response Explanation:
similarity_score: The SSIM score (range from -1 to 1), where 1 means the images are identical, and lower values indicate dissimilarity.
# Notes
The model used for face detection (YOLOv8) is a general object detection model trained to detect multiple objects, including faces. It may not always be perfect for face-only detection in some images, but it works well for most cases.
If the first image is a PDF, only the first page will be processed.
# Contributing
Feel free to fork the repository, submit issues, and create pull requests for improvements.



# Acknowledgments
YOLOv8 by Ultralytics: For object detection.
Poppler: For PDF to image conversion.
Flask: For building the web API.
