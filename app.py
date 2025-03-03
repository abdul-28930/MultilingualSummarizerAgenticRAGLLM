import streamlit as st
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
import torch
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import tempfile
import spacy
import logging
from summary_evaluation import SummaryEvaluator
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure environment variables for better downloads
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = "1"

# Load environment variables
load_dotenv(override=True)

# Initialize summary evaluator
evaluator = SummaryEvaluator()

def initialize_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return OpenAI(api_key=api_key)
    return None

# Initialize spaCy with error handling
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Successfully loaded spaCy model")
except OSError:
    logger.error("SpaCy model not found. Installing...")
    os.system("python -m spacy download en_core_web_sm")
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("Successfully installed and loaded spaCy model")
    except Exception as e:
        logger.error(f"Failed to load spaCy model: {str(e)}")
        st.error("Error loading language model. Please contact support.")

# Lazy load RAG summarization to handle import errors
def get_rag_summarization():
    try:
        from agentrag.ragagent import agentic_rag_summarization
        return agentic_rag_summarization
    except ImportError as e:
        logger.error(f"Failed to import RAG summarization: {str(e)}")
        st.error("Error loading summarization module. Some features may be unavailable.")
        return None

# Initialize Whisper models and processors
@st.cache_resource
def load_whisper_models():
    try:
        # Load Arabic model and processor
        arabic_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        arabic_model = WhisperForConditionalGeneration.from_pretrained("humbleakh/whisper-small-arabic")
        arabic_processor.tokenizer.set_prefix_tokens(language="arabic", task="transcribe")
        
        # Load Hindi model and processor
        hindi_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        hindi_model = WhisperForConditionalGeneration.from_pretrained("humbleakh/whisper-small-hindi")
        hindi_processor.tokenizer.set_prefix_tokens(language="hindi", task="transcribe")
        
        return {
            "arabic": {"model": arabic_model, "processor": arabic_processor},
            "hindi": {"model": hindi_model, "processor": hindi_processor}
        }
    except Exception as e:
        logger.error(f"Error loading Whisper models: {str(e)}")
        st.error("Error loading transcription models. Please contact support.")

@st.cache_resource
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
        st.error("Error loading transcription pipeline. Please contact support.")

def transcribe_audio(audio_file, language):
    try:
        pipe = load_whisper_pipeline(language)
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

def display_evaluation_metrics(metrics, title="Summary Evaluation Metrics"):
    """Display evaluation metrics in a nice format using Streamlit."""
    if not metrics:
        st.warning("No evaluation metrics available")
        return

    st.subheader(title)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Calculate overall score
    overall_score = evaluator.get_overall_score(metrics)
    
    # Display metrics in a simpler format if plotly is not available
    try:
        import plotly.graph_objects as go
        
        # Display overall score with a gauge chart
        with col1:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = overall_score * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Overall Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgray"},
                        {'range': [33, 66], 'color': "gray"},
                        {'range': [66, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        with col1:
            st.metric("Overall Score", f"{overall_score * 100:.1f}%")
    
    # Display individual metrics
    with col2:
        st.write("Individual Metrics:")
        metrics_df = {
            "Metric": [],
            "Score": []
        }
        
        metric_names = {
            'bleu': 'BLEU',
            'semantic_similarity': 'Semantic Sim.',
            'bert_score': 'BERTScore'
        }
        
        for metric, value in metrics.items():
            if metric != "overall_score":
                metrics_df["Metric"].append(metric_names.get(metric, metric))
                metrics_df["Score"].append(f"{value:.4f}")
        
        st.dataframe(metrics_df, hide_index=True)

def main():
    st.title("Multilingual Video Analysis System")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "transcription" not in st.session_state:
        st.session_state.transcription = None
    if "summaries" not in st.session_state:
        st.session_state.summaries = None
    if "extractive_summary" not in st.session_state:
        st.session_state.extractive_summary = None
    if "abstractive_summary" not in st.session_state:
        st.session_state.abstractive_summary = None
    if "evaluation_metrics" not in st.session_state:
        st.session_state.evaluation_metrics = {}
    if "translated_extractive" not in st.session_state:
        st.session_state.translated_extractive = None
    if "translated_abstractive" not in st.session_state:
        st.session_state.translated_abstractive = None
    if "error" not in st.session_state:
        st.session_state.error = None
    
    # Check OpenAI API key
    client = initialize_openai_client()
    if not client:
        st.sidebar.error("OpenAI API key not found in .env file!")
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Transcribe", "Summarize", "Chat"])

    if page == "Transcribe":
        st.header("Upload Video/Audio")
        
        # Language selection
        language = st.selectbox(
            "Select Language",
            ["arabic", "hindi"],
            format_func=lambda x: x.capitalize()
        )
        
        uploaded_file = st.file_uploader("Choose a file", type=['mp4', 'wav', 'mp3'])
        
        if uploaded_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.'+uploaded_file.name.split('.')[-1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                with st.spinner(f"Transcribing {language.capitalize()} audio..."):
                    transcription = transcribe_audio(tmp_file_path, language)
                    st.session_state.transcription = transcription
                    st.success("Transcription completed!")
                    
                st.subheader("Transcription")
                st.text_area("Transcribed Text", st.session_state.transcription, height=300)
                
                # Clean up temp file
                os.unlink(tmp_file_path)
            except Exception as e:
                logger.error(f"Error during transcription: {str(e)}")
                st.error(f"An error occurred during transcription: {str(e)}")

    elif page == "Summarize":
        if st.session_state.transcription:
            st.header("Summarization Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Generate Extractive Summary"):
                    client = initialize_openai_client()
                    if client:
                        with st.spinner("Generating extractive summary..."):
                            st.session_state.extractive_summary = get_extractive_summary(
                                st.session_state.transcription, client
                            )
                            # Evaluate extractive summary
                            if st.session_state.extractive_summary:
                                metrics = evaluator.evaluate_summary(
                                    st.session_state.transcription,
                                    st.session_state.extractive_summary
                                )
                                st.session_state.evaluation_metrics['extractive'] = metrics
                    else:
                        st.error("Please check your OpenAI API key in .env file")
            
            with col2:
                if st.button("Generate Abstractive Summary"):
                    client = initialize_openai_client()
                    if client:
                        with st.spinner("Generating abstractive summary..."):
                            st.session_state.abstractive_summary = get_abstractive_summary(
                                st.session_state.transcription, client
                            )
                            # Evaluate abstractive summary
                            if st.session_state.abstractive_summary:
                                metrics = evaluator.evaluate_summary(
                                    st.session_state.transcription,
                                    st.session_state.abstractive_summary
                                )
                                st.session_state.evaluation_metrics['abstractive'] = metrics
                    else:
                        st.error("Please check your OpenAI API key in .env file")
            
            # Display summaries and their evaluations
            if st.session_state.extractive_summary:
                st.subheader("Extractive Summary")
                st.write(st.session_state.extractive_summary)
                
                # Add translation option for extractive summary
                languages = ["English", "Arabic", "Hindi", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Russian"]
                col_lang1, col_trans1 = st.columns([2, 1])
                with col_lang1:
                    target_lang_extractive = st.selectbox("Translate extractive summary to:", languages, key="extractive_lang")
                with col_trans1:
                    if st.button("Translate", key="translate_extractive"):
                        client = initialize_openai_client()
                        if client:
                            with st.spinner(f"Translating to {target_lang_extractive}..."):
                                st.session_state.translated_extractive = translate_text(
                                    st.session_state.extractive_summary,
                                    target_lang_extractive,
                                    client
                                )
                
                if st.session_state.translated_extractive:
                    st.subheader(f"Translated Extractive Summary ({target_lang_extractive})")
                    st.write(st.session_state.translated_extractive)
                
                if 'extractive' in st.session_state.evaluation_metrics:
                    display_evaluation_metrics(
                        st.session_state.evaluation_metrics['extractive'],
                        "Extractive Summary Evaluation"
                    )
            
            if st.session_state.abstractive_summary:
                st.subheader("Abstractive Summary")
                st.write(st.session_state.abstractive_summary)
                
                # Add translation option for abstractive summary
                col_lang2, col_trans2 = st.columns([2, 1])
                with col_lang2:
                    target_lang_abstractive = st.selectbox("Translate abstractive summary to:", languages, key="abstractive_lang")
                with col_trans2:
                    if st.button("Translate", key="translate_abstractive"):
                        client = initialize_openai_client()
                        if client:
                            with st.spinner(f"Translating to {target_lang_abstractive}..."):
                                st.session_state.translated_abstractive = translate_text(
                                    st.session_state.abstractive_summary,
                                    target_lang_abstractive,
                                    client
                                )
                
                if st.session_state.translated_abstractive:
                    st.subheader(f"Translated Abstractive Summary ({target_lang_abstractive})")
                    st.write(st.session_state.translated_abstractive)
                
                if 'abstractive' in st.session_state.evaluation_metrics:
                    display_evaluation_metrics(
                        st.session_state.evaluation_metrics['abstractive'],
                        "Abstractive Summary Evaluation"
                    )
            
            # Original RAG summarization
            if not st.session_state.summaries:
                rag_summarize = get_rag_summarization()
                if rag_summarize:
                    try:
                        with st.spinner("Generating RAG summaries..."):
                            st.session_state.summaries = rag_summarize(st.session_state.transcription)
                        
                        st.subheader("RAG Analysis")
                        st.write("**Keywords:**", ", ".join(st.session_state.summaries["keywords"]))
                        st.write("**Entities:**", ", ".join(st.session_state.summaries["entities"]))
                        
                        with st.expander("View RAG Summaries"):
                            st.write("**Pegasus Summary:**")
                            st.write(st.session_state.summaries["pegasus_summary"])
                            
                            st.write("**BERTSUM Summary:**")
                            st.write(st.session_state.summaries["bertsum_summary"])
                            
                            st.write("**TextRank Summary:**")
                            st.write(st.session_state.summaries["textrank_summary"])
                    except Exception as e:
                        logger.error(f"Error during summarization: {str(e)}")
                        st.error(f"An error occurred during summarization: {str(e)}")
        else:
            st.warning("Please transcribe a video/audio file first!")

    elif page == "Chat":
        if st.session_state.transcription:
            st.header("Chat about the Content")
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about the content"):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)

                try:
                    # Prepare context for the assistant
                    context = f"""
                    Here is the transcribed content: {st.session_state.transcription}
                    
                    If available, here are the summaries:
                    {st.session_state.summaries if st.session_state.summaries else 'No summaries available'}
                    
                    Please answer the following question about this content: {prompt}
                    """
                    
                    # Add system message with context
                    messages = [
                        {"role": "system", "content": context}
                    ] + st.session_state.messages
                    
                    # Get and display assistant response
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = get_chat_response(messages)
                            st.write(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    logger.error(f"Error during chat: {str(e)}")
                    st.error(f"An error occurred during chat: {str(e)}")
        else:
            st.warning("Please transcribe a video/audio file first!")

if __name__ == "__main__":
    main()
