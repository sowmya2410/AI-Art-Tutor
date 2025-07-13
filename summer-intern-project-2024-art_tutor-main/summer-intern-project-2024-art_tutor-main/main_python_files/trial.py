import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import gradio as gr
import random
from flask import Flask, request, render_template, redirect, url_for, jsonify
import sqlite3
from datetime import datetime
from flask_cors import CORS
import threading
import os 
import cv2
import json
import base64
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

import numpy as np
import matplotlib.pyplot as plt

from fuzzywuzzy import process
#import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
#import gradio as gr
# Initialize the Flask app

trial = Flask(__name__, template_folder='templates')
CORS(trial)
trial.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load the pre-trained model and tokenizer for art review
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the BlenderBot model and tokenizer
chatbot_model_name = "facebook/blenderbot-400M-distill"
chatbot_model = BlenderbotForConditionalGeneration.from_pretrained(chatbot_model_name)
chatbot_tokenizer = BlenderbotTokenizer.from_pretrained(chatbot_model_name)

# Load the pre-trained BERT model and tokenizer
qa_model_name = "bert-base-uncased"
qa_bert_model = BertModel.from_pretrained(qa_model_name)
qa_tokenizer = BertTokenizer.from_pretrained(qa_model_name)

# Initialize conversation history for the chatbot
conversation_history = []

# Function to generate caption for an art image
def generate_caption(image):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Simulate AI-generated feedback based on the user description
def generate_feedback(description):
    feedback_templates = [
        "This piece demonstrates a strong {}. However, to improve, consider enhancing the {}.",
        "The use of {} in this artwork is excellent. To elevate this piece, you might want to focus on improving the {}.",
        "Great work on the {}! To take this to the next level, consider adding more {}.",
        "The {} in this artwork is well-executed. For further improvement, pay attention to the {}.",
        "Your approach to {} is commendable. For a more refined outcome, try working on the {}."
    ]

    elements = [
        "composition", "color balance", "contrast", "lighting", "texture", "depth",
        "shading", "line work", "form", "perspective", "symmetry", "proportion", "realism",
        "abstract elements", "movement", "pattern", "rhythm", "space", "scale", "structure",
        "background integration", "foreground emphasis"
    ]
    improvements = [
        "details", "dynamic range", "harmony", "focus", "perspective", "proportions",
        "brushwork", "layering", "balance", "emphasis", "unity", "variety", "repetition",
        "gradation", "transitions", "sharpness", "softness", "color palette", "edge quality",
        "visual interest", "negative space usage", "depth perception"
    ]

    feedback_parts = []
    while len(feedback_parts) < 5:
        template = random.choice(feedback_templates)
        element = random.choice(elements)
        improvement = random.choice(improvements)
        feedback_parts.append(template.format(element, improvement))
        elements.remove(element)
        improvements.remove(improvement)

    detailed_feedback = " ".join(feedback_parts)
    return detailed_feedback

# Setting up the Gradio interface for art review
def review_art(image, description):
    if not description:
        caption = generate_caption(image)
        description = caption
    feedback = generate_feedback(description)
    review = f"Art Review: {description}\n\nAI Feedback: {feedback}"
    return review

iface1 = gr.Interface(
    fn=review_art,
    inputs=[
        gr.Image(type="pil", label="Upload your art"),
        gr.Textbox(lines=4, placeholder="Describe your art...", label="Art Description")
    ],
    outputs="text",
    title="Artistic Review Tool – AI Insights for Your Masterpieces.",
    description="Upload an art image and get an AI-generated review and feedback.",
)


@trial.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@trial.route('/art_reviewer')
def art_reviewer():
    return render_template('art_reviewer.html')

@trial.route('/chatbot', methods=['POST'])
def handle_prompt():
    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data['prompt']
    
    # Create conversation history string
    history = "\n".join(conversation_history[-10:])  # Limit history to last 10 exchanges

    # Tokenize the input text and history
    input_ids = chatbot_tokenizer(history + "\n" + input_text, return_tensors="pt")['input_ids'][0]
    
    # Ensure the length does not exceed the maximum
    max_length = chatbot_tokenizer.model_max_length
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
    
    # Create a new tensor with the truncated input_ids
    inputs = {'input_ids': input_ids.unsqueeze(0)}
    
    # Generate the response from the model
    outputs = chatbot_model.generate(inputs['input_ids'], max_length=1000, num_beams=5, early_stopping=True)

    # Decode the response
    response = chatbot_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Add interaction to conversation history
    conversation_history.append(f"User: {input_text}")
    conversation_history.append(f"Bot: {response}")

    return response


qa_pairs = {
   "List out the drawing techniques?": "1. Line Drawing Techniques\n2. Shading Techniques\n3. Rendering Techniques\n4. Perspective Techniques\n5. Mixed Media Techniques",
    "What are the basic principles of shading?": "1. Light Source\n2. Cast Shadow\n3. Form Shadow\n4. Highlight\n5. Reflected Light",
    "Name some common perspective techniques in drawing": "1. One-Point Perspective\n2. Two-Point Perspective\n3. Three-Point Perspective\n4. Atmospheric Perspective\n5. Foreshortening",
    "give the details of Line Drawing Techniques": """Contour Drawing\n
- Definition: Contour drawing involves drawing the outline of an object or figure without lifting the drawing tool. It emphasizes the edges and contours to create a sense of form and volume.\n
- Purpose: Helps in understanding the structure and proportions of subjects. Enhances hand-eye coordination and observational skills.\n
Gesture Drawing\n
- Definition: Gesture drawing captures the essence and movement of a subject using quick, expressive lines. It focuses on capturing action, energy, and fluidity rather than details.\n
- Purpose: Improves understanding of movement and dynamics. Develops spontaneity and the ability to capture poses quickly.\n
Cross-Contour Drawing\n
- Definition: Cross-contour drawing involves drawing lines that follow the contours of the form, emphasizing its three-dimensional structure. Lines wrap around the object, indicating its volume.\n
- Purpose: Enhances understanding of form and solidity. Helps in depicting surfaces and textures realistically.\n
""",
    
}



# Function to get BERT embeddings for a given text
def get_bert_embedding(text):
    inputs = qa_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = qa_bert_model(**inputs)  # Use qa_bert_model here
    return outputs.last_hidden_state.mean(dim=1)  # Mean pooling to get a single vector

# Precompute embeddings for predefined questions
predefined_questions = list(qa_pairs.keys())
predefined_embeddings = [get_bert_embedding(question) for question in predefined_questions]

# Function to find the best match for the user's question using fuzzywuzzy and BERT
def find_best_match(user_question, fuzzy_threshold=70, similarity_threshold=0.5):
    # First use fuzzywuzzy to find a match
    best_fuzzy_match, fuzzy_score = process.extractOne(user_question, predefined_questions)

    # If fuzzy match is good enough (score above threshold), use it
    if fuzzy_score >= fuzzy_threshold:
        # Check BERT similarity to ensure it’s a relevant match
        user_embedding = get_bert_embedding(user_question)
        best_match_embedding = get_bert_embedding(best_fuzzy_match)
        similarity = cosine_similarity(user_embedding, best_match_embedding)[0][0]
        if similarity >= similarity_threshold:
            return best_fuzzy_match, qa_pairs[best_fuzzy_match]
    
    # Otherwise, provide a fallback answer
    return "Sorry, I couldn't find an answer for that.", None

# Define the Gradio interface functions
def answer_question(user_input):
    matched_question, answer = find_best_match(user_input)
    return answer if answer else "Sorry, I couldn't find an answer for that."

def display_answer(question):
    return qa_pairs.get(question, "Sorry, I couldn't find an answer for that.")

# Create the Gradio interface
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## Canvas Queries")
        gr.Markdown("### Ask, Learn, and Explore the World of Creativity!")

        # Add buttons for each predefined question
        with gr.Row():
            buttons = [gr.Button(question, elem_id=question) for question in predefined_questions]
        
        # Textbox and submit button
        with gr.Row():
            textbox = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
            submit_button = gr.Button("Submit")
            answer_output = gr.Textbox(label="Answer", placeholder="The answer will appear here...")

        # Connect the input box and submit button to the answer function
        submit_button.click(answer_question, inputs=textbox, outputs=answer_output)

        # Connect each button to display the corresponding answer
        for button in buttons:
            button.click(display_answer, inputs=button, outputs=answer_output)

    return demo

iface2 = create_interface()

@trial.route('/art_qa')
def art_qa():
    return render_template('art_qa.html')


# Initialize the art progress database
def init_db():
    conn = sqlite3.connect('progress.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            similarity REAL,
            feedback TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()
init_db()

# Image comparison function with motivational feedback
def compare_images(ref_img, user_img):
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    user_gray = cv2.cvtColor(user_img, cv2.COLOR_BGR2GRAY)

    # Resize user image to match reference
    user_gray = cv2.resize(user_gray, (ref_gray.shape[1], ref_gray.shape[0]))

    # Find differences
    diff = cv2.absdiff(ref_gray, user_gray)
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # Calculate similarity percentage
    similarity = 100 - (cv2.countNonZero(thresh) / thresh.size * 100)
    similarity = round(similarity, 2)  # Round to 2 decimal places

    # Generate motivational feedback based on similarity
    feedback = ""
    if similarity > 90:
        feedback = "Excellent work! Your drawing is very close to the reference. Keep up the great effort!"
    elif similarity > 80:
        feedback = "Fantastic! Your drawing is quite similar to the reference. Just a few more tweaks needed!"
    elif similarity > 70:
        feedback = "Very good! Your drawing shows strong resemblance to the reference. Great job!"
    elif similarity > 60:
        feedback = "Good effort! There are some differences, but you're on the right track."
    elif similarity > 50:
        feedback = "Nice try! You're making progress. Pay more attention to the details for better results."
    elif similarity > 40:
        feedback = "Decent attempt. Focus on refining your drawing to better match the reference."
    elif similarity > 30:
        feedback = "Fair effort. There's room for improvement. Keep practicing and adjust your details."
    elif similarity > 20:
        feedback = "You've made a start. Try to focus more on the reference image for better accuracy."
    elif similarity > 10:
        feedback = "Initial effort is seen. Work on aligning your drawing with the reference for improvement."
    else:
        feedback = "Keep practicing! There's a lot of room for improvement. Review the reference image closely."

    return similarity, feedback

# Route for uploading images and calculating progress
@trial.route('/upload', methods=['POST'])
def upload_images():
    title = request.form['title']
    ref_file = request.files['reference_image']
    user_file = request.files['user_image']

    ref_path = os.path.join(trial.config['UPLOAD_FOLDER'], ref_file.filename)
    user_path = os.path.join(trial.config['UPLOAD_FOLDER'], user_file.filename)

    ref_file.save(ref_path)
    user_file.save(user_path)

    ref_img = cv2.imread(ref_path)
    user_img = cv2.imread(user_path)

    similarity, feedback = compare_images(ref_img, user_img)

    # Save the progress to the database
    conn = sqlite3.connect('progress.db')
    cursor = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO progress (title, similarity, feedback, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (title, similarity, feedback, timestamp))
    conn.commit()
    conn.close()

    os.remove(ref_path)
    os.remove(user_path)

    return redirect(url_for('pro'))

@trial.route('/pro')
def pro():
    return render_template('pro.html')

# Route for displaying progress
@trial.route('/progress')
def display_progress():
    conn = sqlite3.connect('progress.db')
    cursor = conn.cursor()
    cursor.execute('SELECT title, similarity, feedback, timestamp FROM progress')
    rows = cursor.fetchall()
    conn.close()

    progress_data = [{'title': row[0], 'similarity': row[1], 'feedback': row[2], 'timestamp': row[3]} for row in rows]
    return render_template('progress.html', progress_data=progress_data)


@trial.route('/gallery')
def gallery():
    return render_template('gallery.html')

if __name__ == '__main__':
    try:
        threading.Thread(target=lambda: iface2.launch(server_name="127.0.0.1", server_port=7861, share=True)).start()
        threading.Thread(target=lambda: iface1.launch(server_name="127.0.0.1", server_port=8081, share=True)).start()
        
        trial.run(debug=True, port=5000)

    except Exception as e:
        print("Error: ", e)

