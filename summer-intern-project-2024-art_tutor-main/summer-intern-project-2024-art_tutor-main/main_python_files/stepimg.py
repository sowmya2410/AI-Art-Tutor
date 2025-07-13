from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

stepimg = Flask(__name__)

# Function to progressively add outlines of the image in steps
def progressive_outline_parts(image, num_steps=5):
    # Convert image to grayscale and apply edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and split them into parts
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Create an empty RGBA (transparent) image for each step
    height, width = image.shape[:2]
    step_images = [np.ones((height, width, 4), dtype=np.uint8) * 255 for _ in range(num_steps)]  # White background, transparent image

    # Create step images with outlines
    for i in range(num_steps):
        # Draw the contours progressively as outlines without colors
        for j in range(i + 1):
            if j < len(contours):
                cv2.drawContours(step_images[i], contours, j, (0, 0, 0, 255), thickness=2)  # Black outline with alpha channel

    return step_images

# Function to convert an image to base64
def convert_image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    img_bytes = BytesIO(buffer)
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    return img_base64

# Route to upload the image and generate steps
@stepimg.route("/upload_image", methods=["POST"])
def upload_image():
    file = request.files['image']
    num_steps = int(request.form.get('num_steps', 3))  # Get num_steps from form data, default to 3
    if not file:
        return jsonify({"error": "No file provided"}), 400

    # Load the image using OpenCV
    npimg = np.fromfile(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Generate step images with user-defined number of steps
    step_images = progressive_outline_parts(img, num_steps=num_steps)

    # Convert step images to base64 for HTML display
    step_images_base64 = [convert_image_to_base64(img) for img in step_images]

    # Return the step images as JSON
    return jsonify({"steps": step_images_base64})


# Main route to display the webpage
@stepimg.route("/")
def steps():
    return render_template("steps.html")

if __name__ == "__main__":
    stepimg.run(debug=True,port=4040)