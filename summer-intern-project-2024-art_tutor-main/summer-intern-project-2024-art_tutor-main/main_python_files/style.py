import os
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

style = Flask(__name__)
style.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load pre-trained style transfer model from TensorFlow Hub
hub_url = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
model = hub.load(hub_url)

def load_image(image_path, max_dim=512):
    """Load an image file and resize it to the maximum dimension."""
    img = Image.open(image_path)
    img = img.convert('RGB')
    
    # Resize image using the updated resampling filter
    img = img.resize((max_dim, int(max_dim * img.height / img.width)), Image.Resampling.LANCZOS)
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def style_transfer(content_image, style_image):
    """Perform style transfer."""
    content_image = tf.convert_to_tensor(content_image, dtype=tf.float32)
    style_image = tf.convert_to_tensor(style_image, dtype=tf.float32)

    stylized_image = model(content_image, style_image)[0]
    return stylized_image

def save_image(img, path):
    """Save the stylized image to the specified path."""
    img = np.squeeze(img)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)

@style.route("/", methods=["GET", "POST"])
def styleimg():
    if request.method == "POST":
        # Get the uploaded files
        content_file = request.files['content_image']
        style_file = request.files['style_image']

        # Save uploaded images
        content_path = os.path.join(style.config['UPLOAD_FOLDER'], content_file.filename)
        style_path = os.path.join(style.config['UPLOAD_FOLDER'], style_file.filename)
        content_file.save(content_path)
        style_file.save(style_path)

        # Load and preprocess images
        content_image = load_image(content_path)
        style_image = load_image(style_path)

        # Perform style transfer
        stylized_image = style_transfer(content_image, style_image)

        # Save stylized image
        stylized_image_path = os.path.join(style.config['UPLOAD_FOLDER'], 'stylized_image.png')
        save_image(stylized_image, stylized_image_path)

        # Redirect to display results
        return render_template('styleimg.html', content_image=content_path, 
                               style_image=style_path, stylized_image=stylized_image_path)
    
    return render_template("styleimg.html")




if __name__ == "__main__":
    style.run(debug=True, port=4000)  # Change to any available port


