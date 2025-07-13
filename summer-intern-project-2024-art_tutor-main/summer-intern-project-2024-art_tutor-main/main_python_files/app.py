from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Load and preprocess image
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (800, 600))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

# Segment image using K-means clustering
def segment_image(image_rgb, n_clusters=3):
    pixels = image_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
    labels = kmeans.labels_
    segmented_image = kmeans.cluster_centers_[labels].reshape(image_rgb.shape).astype(np.uint8)
    return segmented_image, labels.reshape(image_rgb.shape[:2]), kmeans

# Extract layer from segmented image
def extract_layer(image, labels, layer_number):
    mask = (labels == layer_number).astype(np.uint8) * 255
    layer = cv2.bitwise_and(image, image, mask=mask)
    return layer

# Save individual layers as images
def save_layers(image, layers):
    saved_layer_paths = []
    for i, layer in enumerate(layers):
        layer_path = f'static/layer_{i}.png'
        plt.imsave(layer_path, layer)
        saved_layer_paths.append(layer_path)
    return saved_layer_paths

@app.route('/')
def layer():
    return render_template('layer.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No image uploaded", 400

    image_file = request.files['image']
    image_path = os.path.join('static', image_file.filename)
    image_file.save(image_path)

    # Process the image
    image_rgb = load_image(image_path)
    segmented_image, labels, kmeans = segment_image(image_rgb)

    # Extract layers
    layers = [extract_layer(segmented_image, labels, i) for i in range(kmeans.n_clusters)]

    # Save individual layers
    saved_layer_paths = save_layers(image_rgb, layers)

    return {"message": "Image processed successfully.", "layer_paths": saved_layer_paths}

@app.route('/static/<filename>')
def result(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    app.run(debug=True, port=9090)
