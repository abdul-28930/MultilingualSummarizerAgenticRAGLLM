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
import yake
import graphviz
import base64
from io import BytesIO

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

def extract_keywords_yake(transcript, num_keywords=10):
    """Extract keywords using YAKE (single-document keyword extraction)."""
    # Set YAKE parameters
    custom_kw_extractor = yake.KeywordExtractor(
        lan="en",   # Language: English
        n=2,        # Extract up to 2-word phrases (bigram keywords)
        dedupLim=0.9,  # Avoid duplicate similar keywords
        top=num_keywords
    )

    # Extract keywords
    keywords = custom_kw_extractor.extract_keywords(transcript)
    return keywords

def create_keyword_flowchart_graphviz(keywords):
    """Create a flowchart from keywords based on their importance using Graphviz."""
    if not keywords:
        return None

    dot = graphviz.Digraph(comment='Keyword Flowchart')
    dot.attr(rankdir='TB', size='8,8')
    dot.attr('node', shape='box', style='filled', fontname='Arial')

    # Determine max score for coloring
    max_score = max([score for _, score in keywords]) if keywords else 1.0

    for keyword, score in keywords:
        color_intensity = int(255 * (score / max_score))
        color = f"#{255 - color_intensity:02x}{255 - color_intensity:02x}ff"
        dot.node(keyword, label=f"{keyword}\n({score:.4f})", fillcolor=color,
                 fontcolor='white' if color_intensity > 128 else 'black')

    # Connect keywords in order of importance
    sorted_keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
    for i in range(len(sorted_keywords) - 1):
        dot.edge(sorted_keywords[i][0], sorted_keywords[i + 1][0])

    # Render to PNG and convert to base64
    png_data = dot.pipe(format='png')
    encoded_image = base64.b64encode(png_data).decode('utf-8')
    return encoded_image

def create_logical_flow_graphviz(transcript, keywords, original_keywords=None):
    """
    Create a logical keyword flow based on keyword appearance in the text.
    
    Args:
        transcript: The original text transcript
        keywords: The keywords to display in the flowchart (can be translated)
        original_keywords: The original untranslated keywords (used to find positions in the text)
    """
    # If original keywords are provided, use them to find positions in the text
    # Otherwise, use the provided keywords (which might be translated)
    position_keywords = original_keywords if original_keywords else keywords
    
    # Create a mapping from original keywords to translated keywords if both are provided
    keyword_map = {}
    if original_keywords and len(original_keywords) == len(keywords):
        for i in range(len(original_keywords)):
            keyword_map[original_keywords[i][0]] = keywords[i][0]
    
    # Find positions of keywords in the transcript
    keyword_positions = {keyword: transcript.lower().find(keyword.lower()) for keyword, _ in position_keywords}
    keyword_positions = {k: v for k, v in keyword_positions.items() if v != -1}
    sorted_keywords = sorted(keyword_positions.items(), key=lambda x: x[1])
    
    # If no keywords found in the text, return empty image
    if not sorted_keywords:
        return None
    
    dot = graphviz.Digraph(comment='Logical Keyword Flow')
    dot.attr(rankdir='TB', size='8,8')
    dot.attr('node', shape='box', style='filled', fontname='Arial')
    
    max_score = max([score for _, score in keywords]) if keywords else 1.0
    
    for keyword, position in sorted_keywords:
        # If we have a mapping, use the translated keyword for display
        display_keyword = keyword_map.get(keyword, keyword) if keyword_map else keyword
        
        # Find the score for this keyword
        score = next((s for k, s in position_keywords if k == keyword), 0.1)
        
        color_intensity = int(255 * (score / max_score))
        color = f"#99{255 - color_intensity:02x}{255 - color_intensity:02x}"
        
        # Use the translated/display keyword in the node label
        dot.node(display_keyword, label=f"{display_keyword}\n({score:.4f})", fillcolor=color, fontcolor='black')
    
    # Connect nodes using the display keywords
    sorted_display_keywords = []
    for keyword, _ in sorted_keywords:
        display_keyword = keyword_map.get(keyword, keyword) if keyword_map else keyword
        sorted_display_keywords.append(display_keyword)
    
    for i in range(len(sorted_display_keywords) - 1):
        dot.edge(sorted_display_keywords[i], sorted_display_keywords[i + 1])
    
    # Render to PNG and convert to base64
    png_data = dot.pipe(format='png')
    encoded_image = base64.b64encode(png_data).decode('utf-8')
    return encoded_image

def translate_keywords_to_english(keywords, client):
    """Translate a list of keywords to English using OpenAI API."""
    if not keywords:
        return []
    
    try:
        # Extract just the keyword texts
        keyword_texts = [keyword for keyword, _ in keywords]
        
        # Create a prompt for translation
        prompt = f"""Translate the following keywords to English. Keep them concise and maintain their technical meaning:
        
        Keywords:
        {', '.join(keyword_texts)}
        
        Return only the translated keywords in the same order, separated by commas."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a professional translator. Provide accurate translations of keywords to English."},
                     {"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        # Parse the response
        translated_text = response.choices[0].message.content
        translated_keywords = [kw.strip() for kw in translated_text.split(',')]
        
        # Make sure we have the same number of translations as original keywords
        if len(translated_keywords) != len(keywords):
            logger.warning(f"Translation returned {len(translated_keywords)} keywords, but expected {len(keywords)}. Using original keywords.")
            return keywords
        
        # Pair translated keywords with original scores
        translated_keywords_with_scores = [(translated_keywords[i], keywords[i][1]) for i in range(len(keywords))]
        
        return translated_keywords_with_scores
    except Exception as e:
        logger.error(f"Error translating keywords: {str(e)}")
        return keywords  # Return original keywords if translation fails

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

@app.route('/api/flowchart', methods=['POST'])
def api_flowchart():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    num_keywords = data.get('num_keywords', 10)
    
    try:
        # Extract keywords
        original_keywords = extract_keywords_yake(text, num_keywords=num_keywords)
        
        # Initialize OpenAI client for translation
        client = initialize_openai_client()
        
        # Translate keywords to English if not already in English
        if client:
            translated_keywords = translate_keywords_to_english(original_keywords, client)
        else:
            translated_keywords = original_keywords
            logger.warning("OpenAI client not initialized. Using original keywords without translation.")
        
        # Generate flowcharts with translated keywords
        keyword_flowchart = create_keyword_flowchart_graphviz(translated_keywords)
        
        # For logical flow, pass both translated and original keywords
        logical_flowchart = create_logical_flow_graphviz(text, translated_keywords, original_keywords)
        
        return jsonify({
            "keywords": [{"keyword": k, "score": s} for k, s in original_keywords],  # Original keywords for reference
            "translated_keywords": [{"keyword": k, "score": s} for k, s in translated_keywords],  # Translated keywords
            "keyword_flowchart": keyword_flowchart,
            "logical_flowchart": logical_flowchart
        })
    except Exception as e:
        logger.error(f"Error generating flowchart: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
