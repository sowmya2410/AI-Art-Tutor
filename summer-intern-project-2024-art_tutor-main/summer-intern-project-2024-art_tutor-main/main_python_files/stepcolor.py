from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

stepcolor = Flask(__name__)

# Function to load and convert the image to RGB
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Function to preprocess image data for clustering
def preprocess_image_for_steps(image):
    resized_image = cv2.resize(image, (200, 200))
    pixel_data = resized_image.reshape((-1, 3))
    return resized_image, pixel_data

# Function to get dominant colors with labels
def get_dominant_colors_with_labels(pixel_data, k=5):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixel_data)
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    return colors, labels

# Function to generate step-by-step coloring images
def generate_step_by_step_coloring(image, colors, labels, k=5):
    canvas = np.ones_like(image) * 255  # White background
    height, width, _ = image.shape
    labels_reshaped = labels.reshape((height, width))

    step_images = []
    for i in range(k):
        mask = (labels_reshaped == i)
        canvas[mask] = colors[i]

        # Convert the canvas to an image and encode it to base64
        fig, ax = plt.subplots()
        ax.imshow(canvas)
        ax.axis('off')

        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches='tight', pad_inches=0)
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        step_images.append(f"data:image/png;base64,{img_str}")
        plt.close(fig)

    return step_images

# Route for the homepage
@stepcolor.route('/')
def home():
    return render_template('stepcolor.html')

# Route for processing image and generating steps
@stepcolor.route('/generate', methods=['POST'])
def generate_steps():
    file = request.files['image']
    steps = int(request.form['steps'])
    
    if file and steps:
        image_path = os.path.join('static/output', file.filename)
        file.save(image_path)

        # Process the image and generate steps
        image = load_image(image_path)
        resized_image, pixel_data = preprocess_image_for_steps(image)
        colors, labels = get_dominant_colors_with_labels(pixel_data, k=steps)
        step_images = generate_step_by_step_coloring(resized_image, colors, labels, k=steps)

        return jsonify({'steps': step_images})
    
    return jsonify({'error': 'Invalid input'})

# Route for clearing images
@stepcolor.route('/clear', methods=['POST'])
def clear_images():
    folder = 'static/output'  # Modify to the correct folder for uploaded images
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)  # Only remove files, not directories
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"Folder {folder} does not exist")

    return jsonify({'message': 'Images cleared'})

if __name__ == '__main__':
    stepcolor.run(debug=True, threaded=False, port = 8000)
