from flask import Flask, request, jsonify, render_template
import torch
from transformers import GPTNeoForCausalLM, GPT2TokenizerFast

train = Flask(__name__)
"""
# Load the fine-tuned model and tokenizer
model = GPTNeoForCausalLM.from_pretrained("final_model")
tokenizer = GPT2TokenizerFast.from_pretrained("final_model")
"""
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-neo-125M")


# Set the model to evaluation mode
model.eval()

def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

@train.route('/')
def home():
    return render_template('finalcanvas.html')

@train.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if user_message:
        response = generate_response(user_message)
        return jsonify({"response": response})
    return jsonify({"response": "Error: No message provided"})

if __name__ == "__main__":
    train.run(debug=True, port=8888)
