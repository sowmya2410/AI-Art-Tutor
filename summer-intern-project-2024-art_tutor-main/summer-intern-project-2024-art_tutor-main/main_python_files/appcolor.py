from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from transformers import CLIPProcessor, CLIPModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import base64

appcolor = Flask(__name__)
appcolor.config['UPLOAD_FOLDER'] = 'static/uploads/'

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

def get_dominant_colors(image_path, num_colors=10):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize((image.width // 10, image.height // 10))
    image_array = np.array(image)
    pixels = image_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    return dominant_colors

def plot_color_checkboxes(colors):
    num_colors = len(colors)
    box_size = 1.5
    fig, ax = plt.subplots(figsize=(5, num_colors * 0.8 + 1))

    for i, color in enumerate(colors):
        hex_color = rgb_to_hex(color)
        rgb_color = tuple(color)
        rect = patches.Rectangle((0, i), box_size, 0.6, linewidth=0, edgecolor='none', facecolor=hex_color)
        ax.add_patch(rect)
        plt.text(box_size + 0.1, i + 0.3, f'{hex_color} {rgb_color}', ha='left', va='center', fontsize=10, color='black')

    ax.set_xlim(0, box_size * 3)
    ax.set_ylim(0, num_colors)
    ax.axis('off')

    img_buf = BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return img_data

@appcolor.route('/', methods=['GET', 'POST'])
def color():
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = os.path.join(appcolor.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            dominant_colors = get_dominant_colors(filename)
            img_data = plot_color_checkboxes(dominant_colors)
            return render_template('color.html', img_data=img_data)
"""
    return render_template('color.html') #,img_data=None)


@appcolor.route('/upload_and_generate', methods=['POST'])
def upload_and_generate():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400

    if file:
        # Save the uploaded file
        filename = os.path.join(appcolor.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Get dominant colors (Color Palette generation)
        dominant_colors = get_dominant_colors(filename)
        img_data = plot_color_checkboxes(dominant_colors)

        # Get brush and painting operation suggestions
        brush_suggestion = suggest_brush(filename)
        operation_suggestion = suggest_painting_operation(filename)

        # Remove the saved file after processing
        os.remove(filename)

        return jsonify({
            'img_data': img_data,
            'brush_suggestion': brush_suggestion,
            'operation_suggestion': operation_suggestion
        })


# Suggest Brush Type
def suggest_brush(image_path):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    brush_labels = ["fine line", "spray", "airbrush", "watercolor", "hard edge", "soft edge", "dry brush", "ink pen"]
    image = Image.open(image_path)
    inputs = processor(text=brush_labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    max_prob, brush_idx = probs.max(dim=1)
    return brush_labels[brush_idx.item()]

# Suggest Painting Operation
def suggest_painting_operation(image_path):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    operations = ["shading", "highlighting", "blending", "texturing", "outlining", "color blocking", "detailing"]
    image = Image.open(image_path)
    inputs = processor(text=operations, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    max_prob, operation_idx = probs.max(dim=1)
    return operations[operation_idx.item()]

# Route to handle brush suggestion and painting operation
@appcolor.route('/suggest-brush', methods=['POST'])
def suggest_brush_and_operation():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    image_path = os.path.join("temp_image.png")
    image_file.save(image_path)
    
    # Get suggestions
    brush_suggestion = suggest_brush(image_path)
    operation_suggestion = suggest_painting_operation(image_path)

    # Clean up the saved image file
    os.remove(image_path)
    
    return jsonify({
        'brush_suggestion': brush_suggestion,
        'operation_suggestion': operation_suggestion
    })


if __name__ == '__main__':
    os.environ["LOKY_MAX_CPU_COUNT"] = "4"
    appcolor.run(debug=True, port=3000)
