from typing import Dict
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from openai import OpenAI
from supabase import Client, create_client
from dotenv import load_dotenv
import os

from openai_assistant import OpenAIAssistantDeps

# Load environment variables
load_dotenv()

# Path to fine-tuned models
FINETUNE_PATH = "../FineTune"

# Language model paths - using fine-tuned models
LANGUAGE_MODELS = {
    'en': f"{FINETUNE_PATH}/whisper-small-english",
    'hi': f"{FINETUNE_PATH}/whisper-small-hi",
    'ar': f"{FINETUNE_PATH}/whisper-small-arabic",
    'de': f"{FINETUNE_PATH}/whisper-small-german",
    'es': f"{FINETUNE_PATH}/whisper-small-spanish",
    'fr': f"{FINETUNE_PATH}/whisper-small-french",
    'ja': f"{FINETUNE_PATH}/whisper-small-japanese",
    'ru': f"{FINETUNE_PATH}/whisper-small-russian",
    'ta': f"{FINETUNE_PATH}/whisper-small-tamil",
    'tr': f"{FINETUNE_PATH}/whisper-small-turkish"
}

# OpenAI model configurations
OPENAI_MODELS = {
    'chat': os.getenv('OPENAI_CHAT_MODEL', 'gpt-4'),
    'embedding': os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
}

async def initialize_assistant() -> OpenAIAssistantDeps:
    """Initialize the OpenAI assistant with fine-tuned Whisper models."""
    
    # Initialize OpenAI client
    openai_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Initialize Supabase client
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY")
    )
    
    # Load fine-tuned Whisper models and processors
    whisper_models: Dict[str, torch.nn.Module] = {}
    whisper_processors: Dict[str, WhisperProcessor] = {}
    
    for lang_code, model_path in LANGUAGE_MODELS.items():
        try:
            # Load the fine-tuned model for each language
            model = WhisperForConditionalGeneration.from_pretrained(model_path)
            processor = WhisperProcessor.from_pretrained(model_path)
            
            whisper_models[lang_code] = model
            whisper_processors[lang_code] = processor
            print(f"Loaded fine-tuned model for {lang_code} from {model_path}")
        except Exception as e:
            print(f"Error loading model for {lang_code}: {e}")
    
    # Create and return dependencies with OpenAI models configuration
    deps = OpenAIAssistantDeps(
        supabase=supabase,
        openai_client=openai_client,
        whisper_models=whisper_models,
        whisper_processors=whisper_processors,
        openai_models=OPENAI_MODELS
    )
    
    return deps 