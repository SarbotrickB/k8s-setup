#This is a sample file that demonstrates how we might create a web application to upload documents, generate embeddings using a pre-trained model, and calculate cosine similarity. 
#This example uses Flask for the web framework and a pre-trained model from Hugging Face's transformers library for generating embeddings.
# We can create a docker image to do this task, and deploy this application

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load pre-trained model and tokenizer from Hugging Face
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def calculate_similarity(vec1, vec2):
    return cosine_similarity(vec1, vec2)[0][0]

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Read file content
        content = file.read().decode('utf-8')
        # Generate embedding
        embedding = get_embedding(content)
        return jsonify({'embedding': embedding.tolist()}), 200

@app.route('/compare', methods=['POST'])
def compare_files():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Both file1 and file2 are required'}), 400
    file1 = request.files['file1']
    file2 = request.files['file2']
    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'Both files must be selected'}), 400
    if file1 and file2:
        # Read file contents
        content1 = file1.read().decode('utf-8')
        content2 = file2.read().decode('utf-8')
        # Generate embeddings
        embedding1 = get_embedding(content1)
        embedding2 = get_embedding(content2)
        # Calculate similarity
        similarity = calculate_similarity(embedding1, embedding2)
        return jsonify({'similarity': similarity}), 200

@app.route('/')
def index():
    return "Document Similarity Service"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
