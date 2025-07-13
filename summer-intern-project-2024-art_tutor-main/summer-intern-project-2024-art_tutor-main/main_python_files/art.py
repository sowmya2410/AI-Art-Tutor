from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from collections import Counter
import os
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
import os
from flask_cors import CORS


app = Flask(__name__)
CORS(app) 
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static', 'uploads')
os.environ['OMP_NUM_THREADS'] = '1'



# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Helper functions
def load_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

# ... [Include all other functions from your code here, like segment_image, get_colors, suggest_brush, etc.] ...

  # Segment image using K-means clustering
def segment_image(image_rgb, n_clusters=3):
    pixels = image_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(pixels)
    labels = kmeans.labels_
    segmented_image = kmeans.cluster_centers_[labels].reshape(image_rgb.shape).astype(np.uint8)
    return segmented_image, labels.reshape(image_rgb.shape[:2]), kmeans

# Extract layer from segmented image
def extract_layer(image, labels,layer_number):
    mask = (labels == layer_number).astype(np.uint8) * 255
    layer = cv2.bitwise_and(image, image, mask=mask)
    return layer

     

# Progressive outline function
def progressive_outline_parts(image, num_steps=5):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    height, width = image.shape[:2]
    step_images = [np.ones((height, width, 3), dtype=np.uint8) * 255 for _ in range(num_steps)]

    for i in range(num_steps):
        for j in range(i + 1):
            cv2.drawContours(step_images[i], contours, j, (0, 0, 0), thickness=2)

    return step_images

# Extract dominant colors
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

def get_colors(image_path, num_colors=10):
    image = Image.open(image_path).convert('RGB').resize((100, 100))
    pixels = np.array(image).reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init=10).fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    color_counts = Counter(labels)
    sorted_colors = sorted(zip(dominant_colors, color_counts.values()), key=lambda x: x[1], reverse=True)
    return [color for color, _ in sorted_colors]

# Classify colors into foreground and background
def classify_colors(colors):
    background_color = colors[0]
    foreground_colors = colors[1:]
    return foreground_colors, background_color

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Brush suggestion function
brush_labels = ["fine line", "spray", "airbrush", "watercolor", "hard edge", "soft edge", "dry brush", "ink pen"]

def suggest_brush(image_path):
    image = Image.open(image_path)
    inputs = processor(text=brush_labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    max_prob, brush_idx = probs.max(dim=1)
    return brush_labels[brush_idx.item()], max_prob.item()

# Painting operation suggestion function
def suggest_painting_operation(image_path):
    operations = ["highlighting", "blending", "gradient", "color blocking", "detailing"]
    image = Image.open(image_path)
    inputs = processor(text=operations, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    max_prob, operation_idx = probs.max(dim=1)
    return operations[operation_idx.item()], max_prob.item()

# Brush suggestion for individual colors
def suggest_brush_for_color(color):
    color_image = Image.new("RGB", (200, 200), tuple(map(int, color * 255)))
    inputs = processor(text=brush_labels, images=color_image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    max_prob, brush_idx = probs.max(dim=1)
    return brush_labels[brush_idx.item()], max_prob.item()

# Generate step-by-step coloring with brush suggestions
def generate_step_by_step_coloring(image, colors, labels, k=5):
    # Initialize a white canvas
    canvas = np.ones_like(image) * 255  # Create a blank white canvas
    height, width, _ = image.shape
    labels_reshaped = labels.reshape((height, width))  # Reshape labels to match image dimensions

    step_images = []  # To store images for each step
    brush_suggestions = []  # To store brush suggestions for each step

    for i in range(k):
        # Create a mask for the current color cluster
        mask = (labels_reshaped == i)
        color_rgb = (colors[i] * 255).astype(np.uint8)  # Scale color back to 0-255 for display
        canvas[mask] = color_rgb  # Apply the color to masked areas on the canvas
        
        # Store the canvas copy as a step
        step_images.append(canvas.copy())

        # Get brush suggestion for the current color
        brush_suggestion, _ = suggest_brush_for_color(colors[i])
        brush_suggestions.append(brush_suggestion)

    return step_images, brush_suggestions

# Generate step-by-step brush suggestions for progressive outlines
def generate_outline_brush_suggestions(outline_images):
    outline_brush_suggestions = []
    
    for outline in outline_images:
        brush_suggestion, _ = suggest_brush_for_color(np.mean(outline, axis=(0, 1)) / 255)
        outline_brush_suggestions.append(brush_suggestion)
        
    return outline_brush_suggestions

def generate_html_report(image_path, foreground_colors, background_color, brush_suggestion, operation_suggestion, outlines, outline_brush_suggestions, layers, step_images, step_brush_suggestions, feedback, filename="arttutor.html"):
    html_content = '''
    <html>
    <head>
        <title>Image Feedback Report</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; }
            .color-box { display: inline-block; width: 100px; height: 100px; margin: 10px; }
            .feedback-section { margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>Image Feedback Report</h1>
        <h3>Color Palette</h3>
    '''

    for color in foreground_colors:
        hex_color = rgb_to_hex(color)
        html_content += f'<div class="color-box" style="background-color:{hex_color};">{hex_color}</div>\n'

    hex_color = rgb_to_hex(background_color)
    html_content += f'<div class="color-box" style="background-color:{hex_color};">{hex_color}</div>\n'

    html_content += f'''
        <h3>Suggested Brush: {brush_suggestion}</h3>
        <h3>Suggested Painting Operation: {operation_suggestion}</h3>
        <h3>Feedback:</h3>
        <ul>
    '''

    for fb in feedback:
        html_content += f"<li>{fb}</li>\n"

    html_content += '<h3>Progressive Outlines:</h3>'

    for idx, (outline, outline_brush) in enumerate(zip(outlines, outline_brush_suggestions)):
        outline_filename = f'outline_step_{idx + 1}.png'
        cv2.imwrite(outline_filename, cv2.cvtColor(outline, cv2.COLOR_RGB2BGR))
        html_content += f'<img src="{outline_filename}" width="300"/>\n'
        html_content += f'<p>Brush Suggestion for Outline Step {idx + 1}: {outline_brush}</p><br>'

    html_content += '<h3>Extracted Layers:</h3>'

    for idx, layer in enumerate(layers):
        layer_filename = f'layer_{idx + 1}.png'
        cv2.imwrite(layer_filename, cv2.cvtColor(layer, cv2.COLOR_RGB2BGR))
        html_content += f'<img src="{layer_filename}" width="300"/>\n'

    html_content += '<h3>Step-by-Step Coloring:</h3>'

    for idx, (step_image, brush_suggestion) in enumerate(zip(step_images, step_brush_suggestions)):
        step_filename = f'step_coloring_{idx + 1}.png'
        cv2.imwrite(step_filename, cv2.cvtColor(step_image, cv2.COLOR_RGB2BGR))
        html_content += f'<img src="{step_filename}" width="300"/>\n'
        html_content += f'<p>Brush Suggestion for Step {idx + 1}: {brush_suggestion}</p>\n'

    html_content += '''
    </body>
    </html>
    '''

    with open(filename, 'w') as f:
        f.write(html_content)
    print(f"HTML report generated as '{filename}'")

# Route for the main page
@app.route('/')
def index():
    return render_template('main.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route to handle file upload and processing
@app.route('/upload', methods=['POST'])


def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        filename = secure_filename(file.filename)
        image_path = os.path.join('static', filename)
        file.save(image_path)

        # Your image processing logic here...
        image_rgb = load_image(image_path)
        
        # Segment the image and get colors
        segmented_image, labels, kmeans = segment_image(image_rgb)
        colors = get_colors(image_path)
        foreground_colors, background_color = classify_colors(colors)
        
        # Generate outlines and suggestions
        outlines = progressive_outline_parts(image_rgb)
        outline_brush_suggestions = generate_outline_brush_suggestions(outlines)

        
        outline_filenames = []
        for idx, outline in enumerate(outlines):
            outline_filename = os.path.join('static', f'outline_step_{idx + 1}.png')
            cv2.imwrite(outline_filename, cv2.cvtColor(outline, cv2.COLOR_RGB2BGR))
            outline_filenames.append(outline_filename)
        # Extract layers
        layers = [extract_layer(image_rgb, labels, i) for i in range(len(outlines))]


        layer_filenames = []
        for idx, layer in enumerate(layers):
            layer_filename = os.path.join('static', f'layer_{idx + 1}.png')
            cv2.imwrite(layer_filename, cv2.cvtColor(layer, cv2.COLOR_RGB2BGR))
            layer_filenames.append(layer_filename)
        
        # Step-by-step coloring suggestions
        step_images, step_brush_suggestions = generate_step_by_step_coloring(image_rgb, colors, labels)

        
        step_images_filenames=[]
        for idx, step_image in enumerate(step_images):
            step_image_filename = os.path.join('static', f'step_coloring_{idx + 1}.png')
            cv2.imwrite(step_image_filename, cv2.cvtColor(step_image, cv2.COLOR_RGB2BGR))
            step_images_filenames.append(step_image_filename)
        
        # Suggestions for brush and operation
        brush_suggestion, _ = suggest_brush(image_path)
        operation_suggestion, _ = suggest_painting_operation(image_path)
        
        # Feedback
        feedback = [
            "Use lighter colors for background.",
            "Use fine brushes for detailing foreground elements.",
            "Apply blending for smoother transitions."
        ]

        # Generate 'arttutor.html' and save it to the static folder
        html_path = os.path.join('static', 'arttutor.html')
        generate_html_report(
            image_path=image_path,
            foreground_colors=foreground_colors,
            background_color=background_color,
            brush_suggestion=brush_suggestion,
            operation_suggestion=operation_suggestion,
            outlines=outlines,
            outline_brush_suggestions=outline_brush_suggestions,
            layers=layers,
            step_images=step_images,
            step_brush_suggestions=step_brush_suggestions,
            feedback=feedback,
            filename=html_path
        )
          # Save your HTML content here

        # Redirect to the generated HTML file
        return redirect(url_for('static', filename='arttutor.html'))

    
   

if __name__ == '__main__':
    app.run(debug=True,port=5014)
