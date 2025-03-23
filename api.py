from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import logging
from dotenv import load_dotenv
import torch
from transformers import pipeline
from openai import OpenAI
from summary_evaluation import SummaryEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure environment variables
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = "1"

# Load environment variables
load_dotenv(override=True)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize summary evaluator
evaluator = SummaryEvaluator()

def initialize_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return OpenAI(api_key=api_key)
    return None

def load_whisper_pipeline(language):
    try:
        model_name = f"humbleakh/whisper-small-{language}"
        return pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
            chunk_length_s=30,
            return_timestamps=True
        )
    except Exception as e:
        logger.error(f"Error loading Whisper pipeline: {str(e)}")
        return None

def transcribe_audio(audio_file, language):
    try:
        pipe = load_whisper_pipeline(language)
        if pipe is None:
            return "Error: Failed to load transcription model"
            
        result = pipe(
            audio_file,
            batch_size=1,
            generate_kwargs={
                "language": language,
                "task": "transcribe",
                "return_timestamps": True
            }
        )
        return result["text"] if isinstance(result, dict) else result
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        return f"Error during transcription: {str(e)}"

def get_extractive_summary(text, client):
    try:
        prompt = f"""Please provide an EXTRACTIVE summary of the following text. 
        An extractive summary should only use the most important sentences from the original text, 
        without modifying them. Select 3-4 key sentences that best represent the main points:

        Text to summarize:
        {text}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a skilled text summarizer. For extractive summaries, only use exact sentences from the original text."},
                     {"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in extractive summarization: {str(e)}")
        return f"Error in summarization: {str(e)}"

def get_abstractive_summary(text, client):
    try:
        prompt = f"""Please provide an ABSTRACTIVE summary of the following text. 
        An abstractive summary should capture the main ideas and present them in a new, concise way, 
        using your own words. Focus on the key points and maintain the core message:

        Text to summarize:
        {text}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a skilled text summarizer. For abstractive summaries, rephrase and synthesize the content into a coherent, concise summary."},
                     {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in abstractive summarization: {str(e)}")
        return f"Error in summarization: {str(e)}"

def translate_text(text, target_language, client):
    """Translate text using OpenAI API."""
    try:
        prompt = f"""Translate the following text to {target_language}. 
        Maintain the meaning and tone of the original text while ensuring it sounds natural in the target language.
        
        Text to translate:
        {text}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a professional translator. Provide accurate and natural-sounding translations."},
                     {"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in translation: {str(e)}")
        return f"Error in translation: {str(e)}"

def get_chat_response(messages):
    try:
        client = initialize_openai_client()
        if client:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        else:
            return "Error: OpenAI client not initialized"
    except Exception as e:
        logger.error(f"Error in getting response: {str(e)}")
        return f"Error in getting response: {str(e)}"

# API Endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    language = request.form.get('language', 'arabic')
    if language not in ['arabic', 'hindi']:
        return jsonify({"error": "Unsupported language"}), 400
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.'+file.filename.split('.')[-1]) as tmp_file:
            file.save(tmp_file.name)
            tmp_file_path = tmp_file.name
        
        transcription = transcribe_audio(tmp_file_path, language)
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return jsonify({"transcription": transcription})
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    summary_type = data.get('type', 'extractive')
    text = data['text']
    
    client = initialize_openai_client()
    if not client:
        return jsonify({"error": "OpenAI API key not found"}), 500
    
    try:
        if summary_type == 'extractive':
            summary = get_extractive_summary(text, client)
        elif summary_type == 'abstractive':
            summary = get_abstractive_summary(text, client)
        else:
            return jsonify({"error": "Invalid summary type"}), 400
        
        # Evaluate summary
        metrics = evaluator.evaluate_summary(text, summary)
        
        return jsonify({
            "summary": summary,
            "metrics": metrics
        })
    except Exception as e:
        logger.error(f"Error during summarization: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def api_translate():
    data = request.json
    if not data or 'text' not in data or 'target_language' not in data:
        return jsonify({"error": "Missing required parameters"}), 400
    
    text = data['text']
    target_language = data['target_language']
    
    client = initialize_openai_client()
    if not client:
        return jsonify({"error": "OpenAI API key not found"}), 500
    
    try:
        translated_text = translate_text(text, target_language, client)
        return jsonify({"translated_text": translated_text})
    except Exception as e:
        logger.error(f"Error during translation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/query', methods=['POST'])
def api_query():
    data = request.json
    if not data or 'prompt' not in data or 'context' not in data:
        return jsonify({"error": "Missing required parameters"}), 400
    
    prompt = data['prompt']
    context = data['context']
    
    # Prepare messages for the chat
    messages = [
        {"role": "system", "content": f"Here is the context: {context}"},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = get_chat_response(messages)
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
